# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | ✅        |

## Reporting a vulnerability

If you discover a security vulnerability, please report it privately rather than
opening a public issue.

- Email the maintainers at **security@your-org.example** with a description of
  the issue, steps to reproduce, and the potential impact.
- Please allow a reasonable time for a fix before any public disclosure.
- We will acknowledge receipt, keep you informed of progress, and credit you in
  the release notes if you wish.

## Handling of secrets

- Configuration is environment-driven. Secrets (`API_KEY`, `POSTGRES_PASSWORD`,
  `ANTHROPIC_API_KEY`) are typed as `SecretStr` and are never written to logs.
- Never commit a real `.env` file. Only `.env.example` (with placeholder values)
  belongs in version control.
- Rotate the default `API_KEY` before deploying outside of local development. In
  `production`, a correct API key is always required for protected endpoints.

## Operational notes

- **Database access** uses the SQLAlchemy ORM exclusively, so queries are
  parameterized by construction (no string-built SQL), mitigating SQL injection.
- **Scraping** respects `robots.txt` by default (`SCRAPE_RESPECT_ROBOTS=true`)
  and rate-limits requests. Review target sites' Terms of Service before
  enabling collection.
- **CORS** origins are configurable; restrict `CORS_ALLOW_ORIGINS` in production
  instead of using `*`.
- **LLM usage** is limited to data cleaning, normalization, and explanation
  generation. The LLM never sees secrets and never decides prices.
