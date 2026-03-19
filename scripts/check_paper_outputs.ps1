param(
    [Parameter(Mandatory = $true)]
    [string]$PaperDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$requiredFiles = @(
    '02_notes/deep_notes.md',
    '03_extraction/paper_brief.md',
    '03_extraction/paper_schema.json',
    '05_slides/slides_outline.md',
    '06_compare/comparison_table.md'
)

$requiredSchemaFields = @(
    'research_question',
    'high_level_algorithmic_idea',
    'training_label_dataset',
    'network_structure',
    'network_output',
    'loss_function',
    'inference_policy',
    'novel_contribution'
)

$briefHeadings = @(
    '## 1. Research Question',
    '## 2. High-Level Algorithmic Idea',
    '## 3. Training Label / Dataset',
    '## 4. Network Structure',
    '## 5. Network Output',
    '## 6. Loss Function',
    '## 7. Inference Policy',
    '## 8. Novel Contribution',
    '## Evidence Pointers',
    '## Unresolved Questions'
)

$deepHeadings = @(
    '## Label Construction',
    '## Output Semantics',
    '## Loss and Optimization Target',
    '## Inference and Search',
    '## Real Novelty',
    '## Ablation Support',
    '## Weaknesses / Ambiguities',
    '## My Interpretation'
)

$fieldHeadings = [ordered]@{
    '## 1. Research Question' = 'research_question'
    '## 2. High-Level Algorithmic Idea' = 'high_level_algorithmic_idea'
    '## 3. Training Label / Dataset' = 'training_label_dataset'
    '## 4. Network Structure' = 'network_structure'
    '## 5. Network Output' = 'network_output'
    '## 6. Loss Function' = 'loss_function'
    '## 7. Inference Policy' = 'inference_policy'
    '## 8. Novel Contribution' = 'novel_contribution'
}

$paperPath = Resolve-Path -LiteralPath $PaperDir -ErrorAction Stop | Select-Object -ExpandProperty Path
$errors = New-Object System.Collections.Generic.List[string]

function Add-CheckError {
    param([string]$Message)
    [void]$script:errors.Add($Message)
}

function Get-RequiredPath {
    param([string]$RelativePath)
    $target = Join-Path $paperPath ($RelativePath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
        Add-CheckError "Missing required file: $RelativePath"
    }
    return $target
}

function Get-SectionBody {
    param(
        [string]$Content,
        [string]$Heading
    )
    $pattern = '(?ms)^{0}\s*\r?\n(?<body>.*?)(?=^##\s|\z)' -f [regex]::Escape($Heading)
    $match = [regex]::Match($Content, $pattern)
    if ($match.Success) {
        return $match.Groups['body'].Value
    }
    return $null
}

function Get-LabelValue {
    param(
        [string]$Section,
        [string]$Label
    )
    if ([string]::IsNullOrEmpty($Section)) {
        return $null
    }
    $pattern = '(?m)^-\s*{0}\s*:\s*(?<value>.*)$' -f [regex]::Escape($Label)
    $match = [regex]::Match($Section, $pattern)
    if ($match.Success) {
        return $match.Groups['value'].Value.Trim()
    }
    return $null
}

function Test-MeaningfulBullet {
    param(
        [string]$Section,
        [string[]]$Disallowed = @()
    )
    if ([string]::IsNullOrEmpty($Section)) {
        return $false
    }
    foreach ($line in ($Section -split "`r?`n")) {
        $trimmed = $line.Trim()
        if (-not $trimmed.StartsWith('- ')) {
            continue
        }
        if ($Disallowed -contains $trimmed) {
            continue
        }
        if ($trimmed.Substring(2).Trim()) {
            return $true
        }
    }
    return $false
}

function Require-Headings {
    param(
        [string]$Path,
        [string[]]$Headings
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }
    $content = Get-Content -Raw -Encoding UTF8 $Path
    foreach ($heading in $Headings) {
        if (-not $content.Contains($heading)) {
            Add-CheckError "Missing heading in $([System.IO.Path]::GetFileName($Path)): $heading"
        }
    }
}

function Check-BriefContent {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }

    $content = Get-Content -Raw -Encoding UTF8 $Path
    $metadata = Get-SectionBody -Content $content -Heading '## Paper Metadata'
    if ($null -eq $metadata) {
        Add-CheckError 'Missing section in paper_brief.md: ## Paper Metadata'
        return
    }

    foreach ($label in @('Title', 'Authors', 'Venue / Year', 'Paper Folder')) {
        $value = Get-LabelValue -Section $metadata -Label $label
        if (-not $value) {
            Add-CheckError "paper_brief.md is missing filled metadata: $label"
        }
    }

    foreach ($heading in $fieldHeadings.Keys) {
        $section = Get-SectionBody -Content $content -Heading $heading
        if ($null -eq $section) {
            continue
        }
        if (-not (Get-LabelValue -Section $section -Label 'Summary')) {
            Add-CheckError "paper_brief.md has empty summary under $heading"
        }
        if (-not (Get-LabelValue -Section $section -Label 'Evidence')) {
            Add-CheckError "paper_brief.md has empty evidence under $heading"
        }
    }

    $evidenceSection = Get-SectionBody -Content $content -Heading '## Evidence Pointers'
    if ($null -eq $evidenceSection) {
        Add-CheckError 'Missing section in paper_brief.md: ## Evidence Pointers'
    }
    elseif (-not (Test-MeaningfulBullet -Section $evidenceSection -Disallowed @('- Section:', '- Figure:', '- Table:', '- Equation:'))) {
        Add-CheckError 'paper_brief.md must contain at least one filled evidence pointer'
    }

    $unresolvedSection = Get-SectionBody -Content $content -Heading '## Unresolved Questions'
    if ($null -eq $unresolvedSection) {
        Add-CheckError 'Missing section in paper_brief.md: ## Unresolved Questions'
    }
    elseif (-not (Test-MeaningfulBullet -Section $unresolvedSection -Disallowed @('- None yet.'))) {
        Add-CheckError 'paper_brief.md must list an unresolved question or explicitly justify why none remain'
    }
}

function Check-DeepNotesContent {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }

    $content = Get-Content -Raw -Encoding UTF8 $Path
    foreach ($heading in $deepHeadings[0..6]) {
        $section = Get-SectionBody -Content $content -Heading $heading
        if ($null -eq $section) {
            continue
        }
        foreach ($label in @('Paper-stated facts', 'Evidence', 'Open questions')) {
            if (-not (Get-LabelValue -Section $section -Label $label)) {
                Add-CheckError "deep_notes.md has empty '$label' under $heading"
            }
        }
    }

    $interpretation = Get-SectionBody -Content $content -Heading '## My Interpretation'
    if ($null -eq $interpretation) {
        return
    }
    foreach ($label in @('Inference', 'Confidence', 'What to verify next')) {
        if (-not (Get-LabelValue -Section $interpretation -Label $label)) {
            Add-CheckError "deep_notes.md has empty '$label' under ## My Interpretation"
        }
    }
}

function Check-Schema {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }

    try {
        $data = Get-Content -Raw -Encoding UTF8 $Path | ConvertFrom-Json
    }
    catch {
        Add-CheckError "Invalid JSON in $([System.IO.Path]::GetFileName($Path)): $($_.Exception.Message)"
        return
    }

    $propertyNames = @($data.PSObject.Properties.Name)
    foreach ($topKey in @('paper_title', 'paper_folder', 'scan_date', 'global_evidence_pointers', 'unresolved_questions')) {
        if ($propertyNames -notcontains $topKey) {
            Add-CheckError "Missing schema key: $topKey"
        }
    }

    $doneFields = 0
    $incompleteFields = 0
    foreach ($field in $requiredSchemaFields) {
        if ($propertyNames -notcontains $field) {
            Add-CheckError "Missing schema field: $field"
            continue
        }

        $payload = $data.$field
        $nestedNames = @($payload.PSObject.Properties.Name)
        foreach ($nestedKey in @('summary', 'evidence', 'status')) {
            if ($nestedNames -notcontains $nestedKey) {
                Add-CheckError "Missing nested key for $field`: $nestedKey"
            }
        }

        $status = [string]$payload.status
        if ($status -notin @('done', 'uncertain', 'pending')) {
            Add-CheckError "Invalid status for $field`: $status"
        }

        $summary = [string]$payload.summary
        if ($status -in @('done', 'uncertain') -and -not $summary.Trim()) {
            Add-CheckError "Schema field must have a non-empty summary when status is $status`: $field"
        }

        $evidence = @()
        if ($null -ne $payload.evidence) {
            $evidence = @($payload.evidence)
        }
        if ($status -eq 'done') {
            if ($evidence.Count -eq 0 -or (@($evidence | Where-Object { -not ([string]$_).Trim() }).Count -gt 0)) {
                Add-CheckError "Schema field marked done must include evidence pointers: $field"
            }
        }

        if ($status -eq 'done') {
            $doneFields += 1
        }
        elseif ($status -in @('uncertain', 'pending')) {
            $incompleteFields += 1
        }
    }

    $globalEvidence = @()
    if ($null -ne $data.global_evidence_pointers) {
        $globalEvidence = @($data.global_evidence_pointers)
    }
    if ($doneFields -gt 0 -and $globalEvidence.Count -eq 0) {
        Add-CheckError 'global_evidence_pointers must not be empty when done fields are present'
    }

    $unresolved = @()
    if ($null -ne $data.unresolved_questions) {
        $unresolved = @($data.unresolved_questions)
    }
    if ($incompleteFields -gt 0 -and $unresolved.Count -eq 0) {
        Add-CheckError 'unresolved_questions must not be empty when uncertain/pending fields remain'
    }
}

function Check-Slides {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }
    $content = Get-Content -Raw -Encoding UTF8 $Path
    $slideCount = ([regex]::Matches($content, '(?m)^## Slide ')).Count
    if ($slideCount -lt 8) {
        Add-CheckError "slides_outline.md must contain at least 8 slides, found $slideCount"
    }
}

function Check-ComparisonTable {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return
    }
    $content = Get-Content -Raw -Encoding UTF8 $Path
    $header = '| Paper | Task | Label Type | Dataset | Encoder | Decoder | Output | Loss | Inference | Novelty | Weakness | Relevance |'
    if (-not $content.Contains($header)) {
        Add-CheckError 'comparison_table.md is missing the standard header row'
    }
    $dataRows = @(
        $content -split "`r?`n" |
        Where-Object {
            $_.StartsWith('|') -and
            (-not $_.StartsWith('|---')) -and
            ($_ -ne $header)
        }
    )
    if ($dataRows.Count -eq 0) {
        Add-CheckError 'comparison_table.md must contain at least one paper row'
    }
}

$requiredPaths = @{}
foreach ($relativePath in $requiredFiles) {
    $requiredPaths[$relativePath] = Get-RequiredPath -RelativePath $relativePath
}

Require-Headings -Path $requiredPaths['03_extraction/paper_brief.md'] -Headings $briefHeadings
Require-Headings -Path $requiredPaths['02_notes/deep_notes.md'] -Headings $deepHeadings
Check-BriefContent -Path $requiredPaths['03_extraction/paper_brief.md']
Check-DeepNotesContent -Path $requiredPaths['02_notes/deep_notes.md']
Check-Schema -Path $requiredPaths['03_extraction/paper_schema.json']
Check-Slides -Path $requiredPaths['05_slides/slides_outline.md']
Check-ComparisonTable -Path $requiredPaths['06_compare/comparison_table.md']

if ($errors.Count -gt 0) {
    foreach ($errorMessage in $errors) {
        Write-Output "[FAIL] $errorMessage"
    }
    exit 1
}

Write-Output '[OK] Paper workspace passes the standard checks.'
