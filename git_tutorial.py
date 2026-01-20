# create a new branch at your current HEAD and switch to it
git switch -c <new-branch-name>

# If you only want to create it (stay on current branch):
git branch <new-branch-name>

# Check you’re clean (optional but recommended)
git status

# commit
git add -A
git commit -m "Describe the change"

# Push the new branch to GitHub and set upstream
git push -u origin <new-branch-name>

# To see each commit’s hash + message (text) AND the actual code changes (diff), use:
git push -u origin <new-branch-name>

# Show last N commits with diff:
git log -p -n 5 --pretty=format:"%h %ad %an%n%s%n" --date=short

# show
git show <commit-hash>

# If you only want the commit list (no code):
git log --oneline --decorate --graph -n 20

