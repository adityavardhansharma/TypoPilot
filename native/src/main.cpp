#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char * kPromptPrefix =
    "Correct only the errors (spelling, spacing, capitalization, and obvious "
    "mistakes) in the text below. Keep the exact wording, meaning, and casual "
    "tone. Do not rephrase, translate, or add anything.\n\nText:";
constexpr const char * kPromptSuffix = "\nCorrected:";

struct Options {
    std::string model_path;
    std::string text;
    std::string input_tsv;
    int threads = std::max(1u, std::thread::hardware_concurrency() / 2);
    int max_tokens = 256;
    int context_size = 1024;
    bool warmup = false;
};

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (const unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"':  out << "\\\""; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out << buf;
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

void usage(const char * program) {
    std::cerr
        << "Usage: " << program << " --model MODEL.gguf [--threads N] "
        << "[--max-tokens N] (--text TEXT | --input-tsv FILE)\n";
}

bool parse_args(int argc, char ** argv, Options & options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) return nullptr;
            return argv[++i];
        };
        if (arg == "--model") {
            const char * value = next(); if (!value) return false;
            options.model_path = value;
        } else if (arg == "--text") {
            const char * value = next(); if (!value) return false;
            options.text = value;
        } else if (arg == "--input-tsv") {
            const char * value = next(); if (!value) return false;
            options.input_tsv = value;
        } else if (arg == "--threads") {
            const char * value = next(); if (!value) return false;
            options.threads = std::max(1, std::atoi(value));
        } else if (arg == "--max-tokens") {
            const char * value = next(); if (!value) return false;
            options.max_tokens = std::max(1, std::atoi(value));
        } else if (arg == "--context") {
            const char * value = next(); if (!value) return false;
            options.context_size = std::max(256, std::atoi(value));
        } else if (arg == "--warmup") {
            options.warmup = true;
        } else {
            return false;
        }
    }
    const bool one_input = !options.text.empty() ^ !options.input_tsv.empty();
    return !options.model_path.empty() && one_input;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    const int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
    if (count <= 0) return {};
    std::vector<llama_token> tokens(count);
    const int result = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, true);
    if (result < 0) return {};
    tokens.resize(result);
    return tokens;
}

struct Result {
    std::string output;
    int prompt_tokens = 0;
    int output_tokens = 0;
    double latency_ms = 0;
};

class Corrector {
public:
    Corrector(const Options & options) : options_(options) {
        ggml_backend_load_all();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        model_ = llama_model_load_from_file(options.model_path.c_str(), model_params);
        if (!model_) throw std::runtime_error("unable to load model");

        vocab_ = llama_model_get_vocab(model_);
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = options.context_size;
        context_params.n_batch = options.context_size;
        context_params.n_ubatch = std::min(options.context_size, 512);
        context_params.n_threads = options.threads;
        context_params.n_threads_batch = options.threads;
        context_params.no_perf = true;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        context_ = llama_init_from_model(model_, context_params);
        if (!context_) throw std::runtime_error("unable to create context");

        prefix_tokens_ = tokenize(vocab_, kPromptPrefix, true);
        if (prefix_tokens_.empty()) throw std::runtime_error("prefix tokenization failed");
        llama_batch prefix_batch = llama_batch_get_one(prefix_tokens_.data(), prefix_tokens_.size());
        if (llama_decode(context_, prefix_batch) != 0) throw std::runtime_error("prefix decode failed");
    }

    ~Corrector() {
        if (context_) llama_free(context_);
        if (model_) llama_model_free(model_);
    }

    Result correct(const std::string & input) {
        const std::string dynamic_prompt = std::string(" ") + input + kPromptSuffix;
        auto dynamic_tokens = tokenize(vocab_, dynamic_prompt, false);
        if (dynamic_tokens.empty()) throw std::runtime_error("tokenization failed");
        const int prompt_token_count = static_cast<int>(prefix_tokens_.size() + dynamic_tokens.size());
        if (prompt_token_count + options_.max_tokens > options_.context_size) {
            throw std::runtime_error("input exceeds configured context");
        }

        llama_memory_t memory = llama_get_memory(context_);
        if (!llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(prefix_tokens_.size()), -1)) {
            llama_memory_clear(memory, false);
            llama_batch prefix_batch = llama_batch_get_one(prefix_tokens_.data(), prefix_tokens_.size());
            if (llama_decode(context_, prefix_batch) != 0) throw std::runtime_error("prefix restore failed");
        }
        llama_sampler * sampler = llama_sampler_init_greedy();
        llama_batch batch = llama_batch_get_one(dynamic_tokens.data(), dynamic_tokens.size());
        std::string output;
        int generated = 0;

        const auto started = std::chrono::steady_clock::now();
        for (int position = static_cast<int>(prefix_tokens_.size()); position + batch.n_tokens < prompt_token_count + options_.max_tokens;) {
            if (llama_decode(context_, batch) != 0) {
                llama_sampler_free(sampler);
                throw std::runtime_error("model decode failed");
            }
            position += batch.n_tokens;
            llama_token token = llama_sampler_sample(sampler, context_, -1);
            if (llama_vocab_is_eog(vocab_, token)) break;

            char small[256];
            int piece_size = llama_token_to_piece(vocab_, token, small, sizeof(small), 0, true);
            if (piece_size < 0) {
                std::vector<char> large(-piece_size);
                piece_size = llama_token_to_piece(vocab_, token, large.data(), large.size(), 0, true);
                if (piece_size < 0) {
                    llama_sampler_free(sampler);
                    throw std::runtime_error("token decoding failed");
                }
                output.append(large.data(), piece_size);
            } else {
                output.append(small, piece_size);
            }
            ++generated;
            batch = llama_batch_get_one(&token, 1);
        }
        const auto stopped = std::chrono::steady_clock::now();
        llama_sampler_free(sampler);

        return {
            trim(output),
            prompt_token_count,
            generated,
            std::chrono::duration<double, std::milli>(stopped - started).count()
        };
    }

private:
    Options options_;
    llama_model * model_ = nullptr;
    llama_context * context_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    std::vector<llama_token> prefix_tokens_;
};

void emit(const std::string & id, const Result & result) {
    std::cout << "{\"id\":\"" << json_escape(id)
              << "\",\"output\":\"" << json_escape(result.output)
              << "\",\"latency_ms\":" << static_cast<long long>(result.latency_ms + 0.5)
              << ",\"prompt_tokens\":" << result.prompt_tokens
              << ",\"output_tokens\":" << result.output_tokens << "}\n";
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    Options options;
    if (!parse_args(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }

    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
    }, nullptr);

    try {
        const auto load_started = std::chrono::steady_clock::now();
        Corrector corrector(options);
        const auto load_stopped = std::chrono::steady_clock::now();
        const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_stopped - load_started).count();
        std::cerr << "[loaded model in " << load_ms << " ms; threads=" << options.threads << "]\n";

        if (options.warmup) (void) corrector.correct("warmup text");

        if (!options.text.empty()) {
            emit("input", corrector.correct(options.text));
            return 0;
        }

        std::ifstream input(options.input_tsv, std::ios::binary);
        if (!input) throw std::runtime_error("unable to open TSV input");
        std::string line;
        while (std::getline(input, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const auto tab = line.find('\t');
            if (tab == std::string::npos) throw std::runtime_error("TSV line has no tab separator");
            emit(line.substr(0, tab), corrector.correct(line.substr(tab + 1)));
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}
