

To publish first set a tag: 
```bash
export TAG=v0.1.14
uv version $TAG
git add .
git commit -m "version bump"
git tag $TAG
git push origin $TAG
```

