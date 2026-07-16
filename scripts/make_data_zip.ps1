# Package the final dataset for upload to the Lightning Studio.
# Produces data_final.zip in the repo root (not committed - data/ is ignored).
$root = Split-Path $PSScriptRoot -Parent
Compress-Archive -Force `
    -Path "$root\data\final" `
    -DestinationPath "$root\data_final.zip"
Get-Item "$root\data_final.zip" | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
