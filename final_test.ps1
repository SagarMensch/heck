$ErrorActionPreference = "Continue"
$log = "final_result.txt"

function Write-Log {
    param([string]$msg)
    Add-Content -Path $log -Value $msg
}

Write-Log "START"

try {
    Write-Log "Testing Python..."
    $test = & python -Version 2>&1
    Write-Log "Python: $test"
    
    Write-Log "Testing torch..."
    $result = & python -Command "import torch; print(torch.__version__)" 2>&1
    Write-Log "Torch: $result"
    
    Write-Log "Testing CUDA..."
    $cuda = & python -Command "import torch; print(torch.cuda.is_available())" 2>&1
    Write-Log "CUDA: $cuda"
    
    Write-Log "Loading model..."
    $load = & python -Command "from transformers import VisionEncoderDecoderModel; m = VisionEncoderDecoderModel.from_pretrained('models/tocr-trained'); print('OK')" 2>&1
    Write-Log "Load: $load"
    
    Write-Log "SUCCESS!"
}
catch {
    Write-Log "ERROR: $_"
}

Write-Log "DONE"