#!/bin/sh
set -eu

htpasswd -bc /etc/nginx/.htpasswd "$REVIEWER_USERNAME" "$REVIEWER_PASSWORD"

exec nginx -g "daemon off;"
