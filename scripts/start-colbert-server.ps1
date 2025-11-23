param (
    [int]$Port = 2017,
    [string]$LogFile = "$env:TEMP\colbert-server.log"
)

Write-Host "Starting ColBERT server on port $Port..."
Write-Host "Logs: $LogFile"

# Kill existing processes running colbert_server.py
# Note: This is a bit broad, matching command line arguments is tricky in pure PS without WMI/CIM sometimes, 
# but we can try.
# Or just rely on the user to stop it.
# For now, let's just start it.

# Start the server in a new window or background
# We use uv run python backend/colbert_server.py
# To keep it running, we can use Start-Process.

$ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
$ProcessInfo.FileName = "uv"
$ProcessInfo.Arguments = "run python backend/colbert_server.py"
$ProcessInfo.RedirectStandardOutput = $true
$ProcessInfo.RedirectStandardError = $true
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.CreateNoWindow = $true

$Process = New-Object System.Diagnostics.Process
$Process.StartInfo = $ProcessInfo

# Redirect output to file
# This is complex in PS async.
# Simpler: Use Start-Process with redirection (requires PS 7+ or specific syntax)
# Or just run it in a new window.

# Let's use Start-Process with redirection to a file via cmd /c
Start-Process -FilePath "cmd.exe" -ArgumentList "/c uv run python backend/colbert_server.py > ""$LogFile"" 2>&1" -WindowStyle Hidden

Write-Host "Server process started in background."
Write-Host "Check logs at $LogFile"
