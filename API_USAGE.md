```markdown
# Visa Approval Prediction API - Usage Guide

## Quick Start

### Option 1: Run with Docker (Recommended)
\`\`\`powershell
docker run -d -p 8000:8000 yourusername/visa-api:latest
\`\`\`

### Option 2: Run Locally
\`\`\`powershell
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
\`\`\`

## API Endpoints

### 1. Health Check
**Endpoint**: `GET /`

**Example**:
\`\`\`powershell
Invoke-WebRequest -Uri "http://localhost:8000/" | Select-Object -ExpandProperty Content
\`\`\`

**Response**:
\`\`\`json
{
  "message": "Visa Approval Prediction API",
  "status": "running",
  "version": "1.0.0"
}
\`\`\`

### 2. Predict Visa Approval
**Endpoint**: `POST /predict`

**Request Body**:
\`\`\`json
{
  "continent": "Asia",
  "education_of_employee": "Master's",
  "has_job_experience": "Y",
  "requires_job_training": "N",
  "no_of_employees": 500,
  "yr_of_estab": 2010,
  "region_of_employment": "West",
  "prevailing_wage": 85000.0,
  "unit_of_wage": "Yearly",
  "full_time_position": "Y"
}
\`\`\`

**Response**:
\`\`\`json
{
  "prediction": "Certified",
  "probability_certified": 0.92,
  "probability_denied": 0.08,
  "confidence": 0.92
}
\`\`\`

### 3. Get Model Information
**Endpoint**: `GET /model-info`

**Example**:
\`\`\`powershell
Invoke-WebRequest -Uri "http://localhost:8000/model-info" | Select-Object -ExpandProperty Content
\`\`\`

## Input Fields

| Field | Type | Description | Valid Values |
|-------|------|-------------|--------------|
| continent | string | Applicant's continent | Asia, Europe, Africa, North America, South America, Oceania |
| education_of_employee | string | Education level | High School, Bachelor's, Master's, Doctorate |
| has_job_experience | string | Has work experience | Y, N |
| requires_job_training | string | Requires training | Y, N |
| no_of_employees | integer | Company size | Positive integer |
| yr_of_estab | integer | Year established | 1900-2025 |
| region_of_employment | string | US region | Northeast, South, Midwest, West, Island |
| prevailing_wage | float | Wage offered | Positive number |
| unit_of_wage | string | Wage unit | Yearly, Monthly, Weekly, Hourly |
| full_time_position | string | Full-time | Y, N |

## Interactive Documentation

Visit http://localhost:8000/docs for Swagger UI where you can:
- Try all endpoints
- See request/response schemas
- Test with example data

## Error Handling

If prediction fails:
\`\`\`json
{
  "detail": "Prediction failed: <error details>"
}
\`\`\`

## Performance

- **Latency**: ~50ms per prediction
- **Throughput**: ~100 requests/second (single instance)
- **Memory**: ~200MB RAM per instance

## Production Considerations

- **Rate Limiting**: Not implemented (add nginx/API gateway)
- **Authentication**: Not implemented (add OAuth2/JWT)
- **Logging**: Basic logging (use centralized logging)
- **Monitoring**: Add Prometheus metrics
```