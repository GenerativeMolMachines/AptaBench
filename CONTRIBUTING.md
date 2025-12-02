# Contributing to AptaBench

We welcome contributions to the project!

## 🎯 How to Help

- 🐛 Report bugs (create an issue)
- 📝 Improve documentation
- 🚀 Add new encoders or models
- 🧪 Write tests

## 🔄 Process

1. **Fork** the repository
2. **Create branch**: `git checkout -b feature/my-feature`
3. **Commit**: `git commit -m "feat: add new feature"`
4. **Push**: `git push origin feature/my-feature`
5. **Create Pull Request**

### Commit Conventions

- `feat:` - new feature
- `fix:` - bug fix
- `docs:` - documentation
- `test:` - tests

## 📝 Code Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Maximum line length: 120 characters
- Add docstrings to functions
- Formatting: `black src/ --line-length 120`

```python
def my_function(sequences, batch_size=32):
    """
    Brief function description.
    
    Parameters
    ----------
    sequences : list
        Parameter description
    batch_size : int
        Batch size
    
    Returns
    -------
    result : array
        Result description
    """
    pass
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## ❓ Questions?

Create an issue with detailed description.

