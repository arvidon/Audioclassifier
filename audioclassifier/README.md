# Audio Classifier frontend

Next.js frontend for the audio classifier. Model inference runs on Modal; Railway hosts only this frontend.

## Run locally

```sh
npm ci
cp .env.example .env
npm run dev
```

## Deploy on Railway

Set the service root directory to `/audioclassifier`. For a local CLI deployment, upload the repository root so Railway can locate that directory.

Set `NEXT_PUBLIC_INFERENCE_URL` to the existing Modal endpoint before building:

```text
https://arvidon--audio-cnn-inference-audioclassifier-inference.modal.run
```

This public variable is embedded into the browser bundle at build time; changing it requires a rebuild. Do not put Modal credentials in public variables.

Configure the Railway service with the Railpack builder, build command `npm run build`, start command `npm run start -- --hostname 0.0.0.0`, and health check path `/`. Next.js uses Railway's injected `PORT`. Generate a Railway domain for public access.

```sh
railway up .. --path-as-root
railway domain
```

For an existing service, link it first with `railway link`. Validate locally with `npm run build`.
