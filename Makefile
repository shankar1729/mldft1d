.PHONY: precommit
precommit:
	pre-commit run --all-files

.PHONY: typecheck
typecheck:
	cd src && mypy -p mldft1d
