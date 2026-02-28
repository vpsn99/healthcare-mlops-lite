param(
  [string]$BaseUrl = "http://127.0.0.1:8000"
)

$body = @{
  age_years = 45
  HEALTHCARE_EXPENSES = 20000
  HEALTHCARE_COVERAGE = 15000
  INCOME = 90000
  encounter_count = 5
  avg_enc_duration_days = 2.5
  active_span_days = 300
  condition_count = 3
  GENDER = "M"
  RACE = "white"
  ETHNICITY = "nonhispanic"
  MARITAL = "M"
  STATE = "Massachusetts"
} | ConvertTo-Json

Write-Host "Health:"
curl.exe "$BaseUrl/health"

Write-Host "`nPredict:"
Invoke-RestMethod -Method Post -Uri "$BaseUrl/predict" -ContentType "application/json" -Body $body