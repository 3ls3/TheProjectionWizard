# ── base image ──────────────────────────────────────────────────────────────
FROM node:20-alpine

# ── setup working directory ────────────────────────────────────────────────
WORKDIR /app

# ── copy package files first for better Docker layer caching ───────────────
COPY package*.json ./

# ── install dependencies ───────────────────────────────────────────────────
RUN npm install

# ── copy application code ──────────────────────────────────────────────────
COPY . .

# ── build the application ──────────────────────────────────────────────────
RUN npm run build

# ── frontend runtime settings ──────────────────────────────────────────────
ENV PORT=3000
EXPOSE $PORT

# ── serve the built application ────────────────────────────────────────────
CMD ["npm", "run", "preview"] 