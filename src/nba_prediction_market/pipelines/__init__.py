"""End-to-end pipelines.

Intentionally free of eager imports: ``build_dataset`` is the documented
``python -m`` entry point, and re-exporting it here would load the module twice
(once as a package attribute, once as ``__main__``), which ``runpy`` warns about.
"""
