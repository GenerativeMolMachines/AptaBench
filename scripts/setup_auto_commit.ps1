# Скрипт для автоматической настройки задачи в Windows Task Scheduler
# Использование: .\scripts\setup_auto_commit.ps1

param(
    [int]$IntervalHours = 2,
    [string]$TaskName = "AutoGitCommitPush"
)

$ErrorActionPreference = "Stop"

# Проверяем права администратора
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "ОШИБКА: Этот скрипт требует прав администратора." -ForegroundColor Red
    Write-Host "Запустите PowerShell от имени администратора и попробуйте снова." -ForegroundColor Yellow
    exit 1
}

$scriptPath = Join-Path $PSScriptRoot "auto_commit_push.ps1"
$repoPath = Split-Path -Parent $PSScriptRoot

# Проверяем, что скрипт существует
if (-not (Test-Path $scriptPath)) {
    Write-Error "Скрипт auto_commit_push.ps1 не найден в $scriptPath"
    exit 1
}

Write-Host "Настройка автоматических коммитов и push..." -ForegroundColor Cyan
Write-Host "Интервал: каждые $IntervalHours час(ов)" -ForegroundColor Cyan
Write-Host "Путь к скрипту: $scriptPath" -ForegroundColor Gray

# Удаляем существующую задачу, если она есть
$existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Write-Host "Удаление существующей задачи..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Создаем действие
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$scriptPath`"" `
    -WorkingDirectory $repoPath

# Создаем триггер (начинается через 1 час, повторяется каждые N часов)
$startTime = (Get-Date).AddHours(1)
$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $startTime `
    -RepetitionInterval (New-TimeSpan -Hours $IntervalHours) `
    -RepetitionDuration (New-TimeSpan -Days 365)

# Настройки задачи
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Создаем принципал (запуск от текущего пользователя)
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

# Регистрируем задачу
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Автоматический коммит и push изменений в Git репозиторий AptaBench" `
        -Force
    
    Write-Host "`nЗадача успешно создана!" -ForegroundColor Green
    Write-Host "Название задачи: $TaskName" -ForegroundColor Green
    Write-Host "Интервал: каждые $IntervalHours час(ов)" -ForegroundColor Green
    Write-Host "`nДля проверки выполните:" -ForegroundColor Cyan
    Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Start-ScheduledTask" -ForegroundColor Gray
    Write-Host "`nДля просмотра статуса:" -ForegroundColor Cyan
    Write-Host "  Get-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host "`nДля удаления задачи:" -ForegroundColor Cyan
    Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor Gray
} catch {
    Write-Error "Ошибка при создании задачи: $_"
    exit 1
}

