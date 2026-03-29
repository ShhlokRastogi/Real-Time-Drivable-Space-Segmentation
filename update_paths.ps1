$files = Get-ChildItem -Path C:\drivableseg -Filter *.py -File
foreach ($file in $files) {
    $content = Get-Content $file.FullName
    if ($content -match 'e:/BEv') {
        $newContent = $content -replace 'e:/BEv', 'c:/drivableseg'
        Set-Content -Path $file.FullName -Value $newContent
        Write-Host "Updated $($file.Name)"
    }
}
