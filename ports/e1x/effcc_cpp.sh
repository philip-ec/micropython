#!/bin/bash
# effcc doesn't accept '-' to read from stdin; convert it to a temp file.
args=("$@")
tmp=""
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "-" ]]; then
        tmp=$(mktemp /tmp/effcc_stdin_XXXXXX.c)
        cat > "$tmp"
        args[$i]="$tmp"
        break
    fi
done
trap '[ -n "$tmp" ] && rm -f "$tmp"' EXIT
exec /home/philip/effcc/bin/effcc "${args[@]}"
