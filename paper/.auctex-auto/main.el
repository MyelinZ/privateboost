;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "twocolumn")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("arxiv" "") ("amsmath" "") ("amssymb" "") ("graphicx" "") ("hyperref" "") ("booktabs" "") ("natbib" "numbers") ("authblk" "")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "href")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "arxiv"
    "amsmath"
    "amssymb"
    "graphicx"
    "hyperref"
    "booktabs"
    "natbib"
    "authblk")
   (LaTeX-add-labels
    "fig:architecture"
    "fig:learning_curves"
    "fig:gain_retention"
    "fig:dropout_resilience"
    "sec:k-anonymous"
    "sec:path-hiding")
   (LaTeX-add-bibliographies
    "references"))
 :latex)

