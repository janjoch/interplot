# Roadmap for future development

- [ ] `fig.add_image`
- [ ] enable `zorder` (supported in `plotly>=5.21.0`)
- [ ] `fig.add_text` more options
    - [ ] font size
    - [ ] rotation
    - [ ] font (?)
    - [ ] fontstyle
        - plotly: supports inline HTML tags
        - plt: only entirely text formatting
        - LaTeX support: on both platforms with $ \dots $, but on plotly it doesn't work in Notebooks...???


# Development workflows


### Sphynx doc

```make html```


### pytest

```pytest```


### build wheels

```python3 -m build```


### upload to PyPi

```twine upload dist/*```

Save access tokens locally in the `~/.pypirc` file (in the user home directory).


### Backup dist files (optional)

Move the files to the local dists backup folder.


### Merge Branch to `main`


### Create GitHub release
