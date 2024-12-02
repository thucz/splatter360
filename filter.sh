  git filter-branch --commit-filter '
    if [ "$GIT_AUTHOR_EMAIL" = "your-old-email@localhost" ];
    then
            GIT_AUTHOR_NAME="Your Correct Name";
            GIT_AUTHOR_EMAIL="your-correct-email@example.com";
            git commit-tree "$@";
    else
            git commit-tree "$@";
    fi' HEAD