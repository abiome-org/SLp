$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\data\project_score')).Path
$url = 'https://cog.sanger.ac.uk/cmp/download/essentiality_matrices.zip'
$first = 100136039L; $last = 149097176L; $count = 32; $width = [math]::Ceiling(($last-$first+1)/$count)
$jobs = for ($i=0; $i -lt $count; $i++) {
    $start=$first+$i*$width; $end=[math]::Min($last,$start+$width-1); $part=Join-Path $root ("part_{0:d2}.bin" -f $i)
    Start-Process curl.exe -ArgumentList @('-L','--fail','--silent','--show-error','--range',"$start-$end",'--output',$part,$url) -PassThru -WindowStyle Hidden
}
$jobs | Wait-Process
if (($jobs | Where-Object ExitCode -ne 0).Count) { throw 'Project Score range download failed' }
$record=Join-Path $root 'corrected_logFCs.record'; $out=[IO.File]::Open($record,[IO.FileMode]::Create)
try { for ($i=0; $i -lt $count; $i++) { $part=Join-Path $root ("part_{0:d2}.bin" -f $i); $bytes=[IO.File]::ReadAllBytes($part); $out.Write($bytes,0,$bytes.Length) } } finally { $out.Dispose() }
if ((Get-Item -LiteralPath $record).Length -ne ($last-$first+1)) { throw 'Project Score record length mismatch' }
for ($i=0; $i -lt $count; $i++) { Remove-Item -LiteralPath (Join-Path $root ("part_{0:d2}.bin" -f $i)) }
