# Uro Server
## Docker Quick Setup

To run the entire stack locally with Docker, use these commands:

### Windows/Mac

Add these lines to your **hosts** file
```
0.0.0.0 vsekai.local
0.0.0.0 uro.v-sekai.cloud
```
then
```
mv frontend/.env.example frontend/.env
mv .env.example .env
docker compose up
```
### Linux
```
# From repository root
mv frontend/.env.example frontend/.env
mv .env.example .env

# Setup hosts file on Linux
# Server runs on 0.0.0.0 port 80 and 443
# Frontend server
sudo bash -c 'echo "0.0.0.0 vsekai.local" >> /etc/hosts'

# Point V-Sekai game to your local server
sudo bash -c 'echo "0.0.0.0 uro.v-sekai.cloud" >> /etc/hosts'

# Run frontend and backend server
docker compose up
```

Server will be available at **http://vsekai.local**

Auto generated root CA will be in `./caddy/data/caddy/pki/authorities/local/root.crt` after you run `docker compose up`.

## Contributing

### Setup

To run the entire stack locally with Docker in **development** mode, use the command:
```
docker compose -f docker-compose.development.yml up
```

By default, the stack uses [Caddy](https://caddyserver.com/) as a reverse proxy and is accessible at http://vsekai.local. You can adjust the values by setting the `ROOT_ORIGIN`, `URL`, and `FRONTEND_URL` environment variables in `.env` and `NEXT_PUBLIC_ORIGIN`, `NEXT_PUBLIC_API_ORIGIN` in `frontend/.env`. Also you will need to set it in `Caddyfile`.

If you want to configure **captcha** for registration, you need to set `TURNSTILE_SECRET_KEY` and `NEXT_PUBLIC_TURNSTILE_SITEKEY` ([Cloudflare Turnstile](https://developers.cloudflare.com/turnstile/get-started/))

Once configured, access the application at:
- http://vsekai.local/
- http://vsekai.local/api/v1/

### OpenAPI Specification

When making changes to Uro, update the OpenAPI specification by running:
```
mix uro.apigen
```
This command generates the OpenAPI specification in `frontend/src/__generated/openapi.json`. The Uro API serves this specification at http://vsekai.local/api/v1/openapi, with documentation available at http://vsekai.local/api/v1/docs.

Once you have updated the OpenAPI specification, to regenerate the client in the frontend (and your editor), run:
```
docker compose -f docker-compose.development.yml up nextjs --build
```
