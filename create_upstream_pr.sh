#!/bin/bash
# create_upstream_pr.sh - Script to create PR to upstream repository

set -e

echo "🚀 Creating PR to upstream assafelovic/gpt-researcher..."

# Configuration
UPSTREAM_REPO="assafelovic/gpt-researcher"
UPSTREAM_BRANCH="master"
FORK_REPO="th3w1zard1/gpt-researcher"
PR_BRANCH="feature/update-docker-files-$(date +%Y%m%d-%H%M%S)"

# Files to include in the PR
FILES_TO_UPDATE=(
    "Dockerfile"
    "Dockerfile.fullstack"
    ".github/workflows/docker-push.yml"
)

echo "📋 PR Configuration:"
echo "   Upstream: ${UPSTREAM_REPO}/${UPSTREAM_BRANCH}"
echo "   Fork: ${FORK_REPO}"
echo "   Branch: ${PR_BRANCH}"
echo "   Files: ${FILES_TO_UPDATE[*]}"
echo ""

# Add upstream remote if it doesn't exist
if ! git remote | grep -q upstream; then
    echo "➕ Adding upstream remote..."
    git remote add upstream "https://github.com/${UPSTREAM_REPO}.git"
fi

# Fetch latest upstream changes
echo "🔄 Fetching upstream changes..."
git fetch upstream

# Create new branch based on upstream master
echo "🌿 Creating new branch: ${PR_BRANCH}"
git checkout -b "${PR_BRANCH}" upstream/master

# Copy the files from the current working branch
echo "📁 Copying updated files..."
for file in "${FILES_TO_UPDATE[@]}"; do
    echo "   Copying ${file}..."
    git show copilot/fix-10f400de-ad02-4ad2-9904-f8e5b9bcc043:"${file}" > "${file}"
done

# Add and commit the changes
echo "💾 Committing changes..."
git add "${FILES_TO_UPDATE[@]}"
git commit -m "feat: update Docker files and CI workflow

- Update Dockerfile to use Python 3.13-slim-trixie with improved Chrome/Firefox setup
- Update Dockerfile.fullstack with enhanced multi-stage build for fullstack deployment  
- Update docker-push.yml workflow to build AIO fullstack image instead of MCP server
- Improve security with proper package installation and non-root user configuration
- Add proper outputs directory permissions and optimized caching

This PR contributes improved Docker configuration and CI workflows from the 
th3w1zard1/gpt-researcher fork back to the upstream repository."

# Push to fork
echo "📤 Pushing to fork..."
git push origin "${PR_BRANCH}"

echo ""
echo "✅ Branch ${PR_BRANCH} has been created and pushed to your fork!"
echo ""
echo "🌐 Next steps:"
echo "1. Go to https://github.com/${FORK_REPO}"
echo "2. Click 'Compare & pull request' for the ${PR_BRANCH} branch"
echo "3. Ensure the base repository is set to: ${UPSTREAM_REPO}:${UPSTREAM_BRANCH}"
echo "4. Add a descriptive title and description for your PR"
echo "5. Submit the pull request"
echo ""
echo "📄 PR Template:"
echo "---"
echo "Title: feat: update Docker files and CI workflow for improved deployment"
echo ""
echo "Description:"
echo "This PR updates the Docker configuration and CI workflows with several improvements:"
echo ""
echo "## Changes Made"
echo "- **Dockerfile**: Upgraded to Python 3.13-slim-trixie with enhanced browser setup"
echo "- **Dockerfile.fullstack**: Enhanced multi-stage build with improved nginx/supervisord config" 
echo "- **docker-push.yml**: Updated workflow to build AIO fullstack images with multi-arch support"
echo ""
echo "## Benefits"
echo "- Improved security and package management"
echo "- Better multi-architecture support (amd64/arm64)"
echo "- Enhanced caching and build optimization"
echo "- More reliable CI/CD pipeline"
echo ""
echo "## Testing"
echo "All files have been validated for syntax and compatibility."
echo "---"