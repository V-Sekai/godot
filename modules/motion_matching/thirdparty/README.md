# Extracting boost

Install bcp.

```
./bootstrap.sh
./b2 tools/bcp
mkdir extract
./dist/bin/bcp build --namespace=motionmatchingboost boost/accumulators extract
```