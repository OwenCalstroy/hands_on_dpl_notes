#!/bin/bash

# 获取当前日期，格式为 YYYY-MM-DD
DATE=$(date +%Y-%m-%d)

# 设置Git仓库的URL
REPO_URL="https://github.com/username/repo.git"

# 添加所有新文件到Git
git add .

# 提交更改，使用当前日期作为提交信息
git commit -m "Added on $DATE"

# 推送更改到GitHub
git push