# Maintainer Notes

## Release Versioning

1. Create a new branch from main.
2. Update the `__version__` in [`dabl/__init__.py`](../dabl/__init__.py).

    For instance

    ```python
    __version__ = "0.2.8-dev"
    ```

    > Include `-dev` suffix to keep local install version suffix.
    >
    > The Github Release Action will use the tag referenced.

3. Submit and merge a PR for the changes.
4. Update the tag locally.

    Once the PR with the new version files is merged.

    ```sh
    git checkout main
    git pull
    git tag M.m.p
    ```

    > Note: `M.m.p` is the version number you just bumped to above.

5. Update the tag remotely on the dabl upstream repo.

    ```sh
    git push --tags # upstream (if that's what you called your upstream git remote)
    ```

6. Make a "Release" on Github.

    > Once this is done, the rules in [`.github/workflows/python-publish.yml`](./.github/workflows/python-publish.yml) will automatically publish the wheels to [pypi](https://pypi.org/project/dabl/).
    > \
    > Note: This may fail if the version number is already published to pypi, in which case start from the beginning with a new patch version.
