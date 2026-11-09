Optional. Regenerates assets/ribbon-light.svg and ribbon-dark.svg from daily commit counts.
  python generate.py daily.json ../../assets
daily.json maps "YYYY-MM-DD" to a commit count. Edit TODAY and START at the top of generate.py to move the window.
Requires numpy. The ribbon is static: it does not update itself.
