#!/bin/bash

# 🚀 AI Policy Query System - Quick Deploy Script

set -e

echo "🚀 Pushing to GitHub for deployment..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not found. Please initialize git first:"
    echo "git init && git add . && git commit -m 'Initial commit'"
    exit 1
fi

# Push to GitHub
git add .
git commit -m "Deploy to production" || true
git push origin main

echo "✅ Code pushed to GitHub!"
echo ""
echo "📋 Next Steps:"
echo "1. Deploy backend to Render: https://render.com"
echo "2. Deploy frontend to Netlify: https://netlify.com"
echo "3. Update API URL in frontend/index.html" 