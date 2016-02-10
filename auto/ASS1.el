(TeX-add-style-hook
 "ASS1"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"
    "amssymb"
    "amsfonts")
   (TeX-add-symbols
    "p"
    "xo"
    "Xo"))
 :latex)

