#!/bin/bash
set -e

SECRETS_DIR="./secrets"
mkdir -p "$SECRETS_DIR"

echo "=== Docker Secrets Initialization ==="
echo

# Exchange API credentials
if [ ! -f "$SECRETS_DIR/exchange_api_key" ]; then
    read -p "Enter Exchange API Key: " API_KEY
    echo -n "$API_KEY" > "$SECRETS_DIR/exchange_api_key"
    chmod 600 "$SECRETS_DIR/exchange_api_key"
    echo "✓ Created exchange_api_key"
else
    echo "✓ exchange_api_key already exists"
fi

if [ ! -f "$SECRETS_DIR/exchange_api_secret" ]; then
    read -p "Enter Exchange API Secret: " API_SECRET
    echo -n "$API_SECRET" > "$SECRETS_DIR/exchange_api_secret"
    chmod 600 "$SECRETS_DIR/exchange_api_secret"
    echo "✓ Created exchange_api_secret"
else
    echo "✓ exchange_api_secret already exists"
fi

# Kill switch secret
if [ ! -f "$SECRETS_DIR/kill_switch_secret" ]; then
    KILL_SWITCH_SECRET=$(openssl rand -base64 32)
    echo -n "$KILL_SWITCH_SECRET" > "$SECRETS_DIR/kill_switch_secret"
    chmod 600 "$SECRETS_DIR/kill_switch_secret"
    echo "✓ Generated kill_switch_secret: $KILL_SWITCH_SECRET"
else
    echo "✓ kill_switch_secret already exists"
fi

# Database password
if [ ! -f "$SECRETS_DIR/db_password" ]; then
    DB_PASSWORD=$(openssl rand -base64 32)
    echo -n "$DB_PASSWORD" > "$SECRETS_DIR/db_password"
    chmod 600 "$SECRETS_DIR/db_password"
    echo "✓ Generated db_password: $DB_PASSWORD"
else
    echo "✓ db_password already exists"
fi

echo
echo "=== Secrets initialized successfully ==="
echo "Secret files created in: $SECRETS_DIR"
echo "File permissions set to 600 (owner read/write only)"
echo
echo "IMPORTANT: Never commit secrets/* to version control!"
