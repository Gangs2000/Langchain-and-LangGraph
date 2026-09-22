---
name: weather-policy
description: Use for current-weather requests. Call the search_weather MCP tool once, then report temperature, condition, humidity, and wind.
---

# Weather policy

1. Use `search_weather` only for a current-weather request.
2. Do not call it more than once per user request.
3. State the city and observation time in the final answer.