FROM node:24-slim AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build  

FROM node:24-slim
WORKDIR /app
# COPY --from=builder /app/.next ./.next
# COPY --from=builder /app/public ./public
# COPY --from=builder /app/package*.json ./
# COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/. .


EXPOSE 3000

ENTRYPOINT [ "npm", "start" ]