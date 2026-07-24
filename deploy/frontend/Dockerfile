FROM node:22-alpine AS applicant-build

WORKDIR /app

COPY applicant/package.json applicant/package-lock.json ./
RUN npm ci

COPY applicant/ .
RUN npm run build

FROM nginx:1.29-alpine AS applicant

COPY --from=applicant-build /app/dist /usr/share/nginx/html
COPY logo_btr.png /usr/share/nginx/html/logo_btr.png
COPY applicant.conf /etc/nginx/conf.d/default.conf

FROM node:22-alpine AS reviewer-build

WORKDIR /app

COPY reviewer/package.json reviewer/package-lock.json ./
RUN npm ci

COPY reviewer/ .
RUN npm run build

FROM nginx:1.29-alpine AS reviewer

RUN apk add --no-cache apache2-utils

COPY --from=reviewer-build /app/dist /usr/share/nginx/html
COPY logo_btr.png /usr/share/nginx/html/logo_btr.png
COPY reviewer.conf /etc/nginx/conf.d/default.conf
COPY reviewer-entrypoint.sh /reviewer-entrypoint.sh

ENTRYPOINT ["sh", "/reviewer-entrypoint.sh"]
