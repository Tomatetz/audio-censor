#!/bin/zsh
cd -- "${0:A:h}"
source .venv/bin/activate
STREAM_CENSOR_TTY="$(tty)"
python web_gui.py

# Finder opens .command files in Terminal. After the local server stops, close
# only the Terminal window containing this exact TTY, leaving other windows
# and tabs untouched.
osascript - "$STREAM_CENSOR_TTY" <<'APPLESCRIPT'
on run argv
	set targetTTY to item 1 of argv
	tell application "Terminal"
		repeat with terminalWindow in windows
			repeat with terminalTab in tabs of terminalWindow
				if tty of terminalTab is targetTTY then
					close terminalWindow
					return
				end if
			end repeat
		end repeat
	end tell
end run
APPLESCRIPT
