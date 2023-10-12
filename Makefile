push:
	rsync -ravhP --filter=":- .gitignore" --filter=":- ~/.gitignore" . gpu:"projects/dumb"