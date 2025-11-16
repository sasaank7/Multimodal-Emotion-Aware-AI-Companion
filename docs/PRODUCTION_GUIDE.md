# Production Deployment Guide

Complete guide for deploying the Emotion-Aware AI Companion in production.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Configuration](#configuration)
3. [Deployment Methods](#deployment-methods)
4. [Security](#security)
5. [Monitoring](#monitoring)
6. [Scaling](#scaling)
7. [Maintenance](#maintenance)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- OS: Ubuntu 20.04+ / Debian 11+ / CentOS 8+

**Recommended:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (optional, for faster inference)

### Software Requirements

- Docker 24.0+
- Docker Compose 2.20+
- Python 3.10+
- PostgreSQL 15+ (or use Docker container)
- Redis 7+ (or use Docker container)
- Nginx (for reverse proxy)

---

## Configuration

### 1. Environment Variables

Create `.env.production`:

```bash
# Database
DB_HOST=postgres
DB_NAME=emotion_ai
DB_USER=emotion_ai_user
DB_PASSWORD=your_secure_password_here

# Redis
REDIS_URL=redis://redis:6379/0

# Security
JWT_SECRET=your_jwt_secret_key_here
SECRET_KEY=your_app_secret_key_here

# Monitoring
GRAFANA_PASSWORD=your_grafana_password_here

# Optional: External Services
HUGGINGFACE_TOKEN=your_hf_token_here
SENTRY_DSN=your_sentry_dsn_here
```

### 2. Application Configuration

Edit `config/production.yaml`:

```yaml
# Update these settings for your environment
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4  # Adjust based on CPU cores

cache:
  redis_url: "redis://redis:6379/0"

database:
  host: "${DB_HOST}"
  port: 5432
  name: "${DB_NAME}"
  user: "${DB_USER}"
  password: "${DB_PASSWORD}"

security:
  ssl:
    enabled: true
    cert_file: "/path/to/cert.pem"
    key_file: "/path/to/key.pem"
```

### 3. SSL/TLS Certificates

For production, use Let's Encrypt:

```bash
# Install certbot
sudo apt-get install certbot

# Get certificates
sudo certbot certonly --standalone -d yourdomain.com -d api.yourdomain.com

# Certificates will be in /etc/letsencrypt/live/yourdomain.com/
```

---

## Deployment Methods

### Method 1: Docker Compose (Recommended)

#### Quick Deploy

```bash
# Clone repository
git clone https://github.com/yourusername/emotion-ai-companion.git
cd emotion-ai-companion

# Copy environment file
cp .env.example .env.production
# Edit .env.production with your values

# Deploy
./scripts/deploy.sh production
```

#### Manual Deploy

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Method 2: Kubernetes

#### Prerequisites

- Kubernetes cluster (v1.25+)
- kubectl configured
- Helm 3+

#### Deploy

```bash
# Create namespace
kubectl create namespace emotion-ai

# Create secrets
kubectl create secret generic emotion-ai-secrets \
  --from-env-file=.env.production \
  -n emotion-ai

# Deploy with Helm
helm install emotion-ai ./k8s/helm-chart \
  -n emotion-ai \
  --values k8s/values-production.yaml

# Check status
kubectl get pods -n emotion-ai
```

### Method 3: Traditional Server

#### Install Dependencies

```bash
# System packages
sudo apt-get update
sudo apt-get install -y python3.10 python3-pip postgresql redis-server nginx

# Python packages
pip3 install -r requirements.txt
```

#### Configure Services

```bash
# PostgreSQL
sudo -u postgres createdb emotion_ai
sudo -u postgres createuser emotion_ai_user

# Nginx
sudo cp nginx/nginx.conf /etc/nginx/sites-available/emotion-ai
sudo ln -s /etc/nginx/sites-available/emotion-ai /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# Systemd service
sudo cp scripts/emotion-ai.service /etc/systemd/system/
sudo systemctl enable emotion-ai
sudo systemctl start emotion-ai
```

---

## Security

### 1. API Authentication

Enable JWT authentication in `config/production.yaml`:

```yaml
api:
  auth:
    enabled: true
    jwt_secret: "${JWT_SECRET}"
    jwt_algorithm: "HS256"
    token_expire_minutes: 60
```

### 2. Rate Limiting

Configure rate limits:

```yaml
api:
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst: 10
```

### 3. CORS Configuration

Set allowed origins:

```yaml
api:
  cors:
    allow_origins:
      - "https://yourdomain.com"
      - "https://app.yourdomain.com"
```

### 4. SSL/TLS

**Nginx configuration:**

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 5. Firewall Rules

```bash
# Allow only necessary ports
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 22/tcp  # SSH
sudo ufw enable
```

---

## Monitoring

### 1. Prometheus & Grafana

Access monitoring dashboards:

- **Grafana**: http://localhost:3000 (default: admin/your_password)
- **Prometheus**: http://localhost:9090

#### Key Metrics to Monitor

- API response time
- Request rate and error rate
- CPU and memory usage
- Database connections
- Cache hit rate
- Model inference time

### 2. Application Logs

```bash
# View logs
docker-compose -f docker-compose.prod.yml logs -f api

# Search logs
docker-compose -f docker-compose.prod.yml logs api | grep ERROR

# Export logs
docker-compose -f docker-compose.prod.yml logs api > api.log
```

### 3. Health Checks

Regular health check endpoints:

```bash
# Application health
curl http://localhost:8000/health

# Detailed status
curl http://localhost:8000/api/v1/status

# Database health
curl http://localhost:8000/api/v1/health/database
```

### 4. Alerts

Configure alerts in `monitoring/alerts.yml`:

```yaml
groups:
  - name: emotion_ai_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: HighMemoryUsage
        expr: memory_usage_percent > 90
        for: 5m
        annotations:
          summary: "Memory usage above 90%"
```

---

## Scaling

### Horizontal Scaling

#### Docker Compose

```bash
# Scale API workers
docker-compose -f docker-compose.prod.yml up -d --scale api=4

# Scale with load balancer
docker-compose -f docker-compose.prod.yml -f docker-compose.scale.yml up -d
```

#### Kubernetes

```bash
# Scale deployment
kubectl scale deployment emotion-ai-api --replicas=4 -n emotion-ai

# Auto-scaling
kubectl autoscale deployment emotion-ai-api \
  --min=2 --max=10 \
  --cpu-percent=70 \
  -n emotion-ai
```

### Vertical Scaling

Update resource limits in `docker-compose.prod.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
```

### Database Scaling

#### Read Replicas

```yaml
services:
  postgres-replica:
    image: postgres:15-alpine
    environment:
      POSTGRES_MASTER_SERVICE_HOST: postgres
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
```

#### Connection Pooling

Use PgBouncer:

```yaml
services:
  pgbouncer:
    image: pgbouncer/pgbouncer:latest
    environment:
      DB_HOST: postgres
      DB_PORT: 5432
      DB_USER: emotion_ai_user
      DB_PASSWORD: ${DB_PASSWORD}
    ports:
      - "6432:6432"
```

---

## Maintenance

### 1. Backups

#### Automated Backups

```bash
# Run backup
./scripts/deploy.sh production backup

# Scheduled backups (cron)
0 2 * * * /path/to/emotion-ai-companion/scripts/deploy.sh production backup
```

#### Manual Backup

```bash
# Database backup
docker-compose -f docker-compose.prod.yml exec postgres \
  pg_dump -U emotion_ai_user emotion_ai > backup_$(date +%Y%m%d).sql

# User data backup
tar -czf user_data_$(date +%Y%m%d).tar.gz data/user_data/
```

#### Restore from Backup

```bash
# Restore database
docker-compose -f docker-compose.prod.yml exec -T postgres \
  psql -U emotion_ai_user emotion_ai < backup_20240115.sql

# Restore user data
tar -xzf user_data_20240115.tar.gz -C data/
```

### 2. Updates

#### Application Updates

```bash
# Pull latest code
git pull origin main

# Rebuild and redeploy
./scripts/deploy.sh production deploy

# Or with zero downtime
docker-compose -f docker-compose.prod.yml up -d --no-deps --build api
```

#### Database Migrations

```bash
# Run migrations
docker-compose -f docker-compose.prod.yml run --rm api \
  python -m alembic upgrade head

# Rollback migration
docker-compose -f docker-compose.prod.yml run --rm api \
  python -m alembic downgrade -1
```

### 3. Log Rotation

Configure log rotation in `/etc/logrotate.d/emotion-ai`:

```
/path/to/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        docker-compose -f /path/to/docker-compose.prod.yml restart api
    endscript
}
```

---

## Troubleshooting

### Common Issues

#### 1. API Not Starting

```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs api

# Common causes:
# - Database not ready
# - Missing environment variables
# - Port already in use

# Solution: Check dependencies
docker-compose -f docker-compose.prod.yml ps
```

#### 2. High Memory Usage

```bash
# Check memory usage
docker stats

# Solution: Enable model quantization in config
# Or increase memory limits
```

#### 3. Slow Response Times

```bash
# Check metrics
curl http://localhost:8000/api/v1/status

# Solutions:
# - Enable Redis caching
# - Use model quantization
# - Scale horizontally
# - Add GPU support
```

#### 4. Database Connection Errors

```bash
# Test database connection
docker-compose -f docker-compose.prod.yml exec api \
  python -c "from src.utils.config_loader import get_config; print('OK')"

# Check PostgreSQL logs
docker-compose -f docker-compose.prod.yml logs postgres
```

### Performance Tuning

#### 1. Model Optimization

```yaml
# config/production.yaml
llm:
  quantization:
    enabled: true
    bits: 4  # Use 4-bit quantization
    compute_dtype: "float16"
```

#### 2. Caching

```yaml
cache:
  enabled: true
  default_ttl: 3600
  model_cache_ttl: 86400
```

#### 3. Worker Tuning

```bash
# Optimal workers = (2 * CPU cores) + 1
# For 4 cores: 9 workers
gunicorn --workers 9 --worker-class uvicorn.workers.UvicornWorker
```

---

## Best Practices

### 1. Security

- ✅ Use strong passwords
- ✅ Enable SSL/TLS
- ✅ Implement rate limiting
- ✅ Regular security updates
- ✅ Use secrets management (Vault, AWS Secrets Manager)
- ✅ Enable CORS only for trusted domains

### 2. Performance

- ✅ Enable caching
- ✅ Use model quantization
- ✅ Implement connection pooling
- ✅ Monitor and optimize slow queries
- ✅ Use CDN for static assets

### 3. Reliability

- ✅ Regular backups
- ✅ Health checks
- ✅ Graceful shutdowns
- ✅ Retry logic
- ✅ Circuit breakers

### 4. Monitoring

- ✅ Set up alerts
- ✅ Monitor key metrics
- ✅ Regular log analysis
- ✅ Performance profiling
- ✅ User analytics

---

## Support

### Getting Help

- **Documentation**: https://docs.yourdomain.com
- **GitHub Issues**: https://github.com/yourusername/emotion-ai/issues
- **Email**: support@yourdomain.com
- **Discord**: https://discord.gg/yourinvite

### Reporting Issues

Include:
1. Environment details
2. Steps to reproduce
3. Error logs
4. Expected vs actual behavior

---

**Last Updated:** 2024-01-15
**Version:** 2.0.0
