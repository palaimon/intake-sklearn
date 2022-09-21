from intake.source.base import DataSource, Schema
import joblib
import fsspec
import sklearn
import re

from . import __version__

class SklearnModelSource(DataSource):
    container = 'python'
    name = 'sklearn'
    version = __version__
    partition_access = False

    def __init__(self, urlpath, storage_options=None, metadata=None):
        """
        Parameters
        ----------

        urlpath: str, location of model pkl file
        Either the absolute or relative path to the file or URL to be
        opened. Some examples:
          - ``{{ CATALOG_DIR }}/models/model.pkl``
          - ``s3://some-bucket/models/model.pkl``
        """
        self._urlpath = urlpath
        self._storage_options = storage_options or {}

        super().__init__(metadata=metadata)


    def _load(self):
        with fsspec.open(self._urlpath, mode='rb', **self._storage_options) as f:
            return f.read()


    def _get_schema(self):
        as_binary = self._load().decode('utf-8', 'ignore')
        has_sklearn_version = re.search(r'_sklearn_version', as_binary)
        s = re.search(r'(_sklearn_version).*(\d\.\d\.\d)', as_binary)

        if has_sklearn_version:
            sklearn_version = s.group(2) if s else None
        else:
            # if no _sklearn_version on pkl file it will ignore
            # the sklearn version validation
            sklearn_version = sklearn.__version__

        self._schema = Schema(
            npartitions=1,
            extra_metadata={
                'sklearn_version':sklearn_version
            }
        )
        return self._schema


    def read(self):
        self._load_metadata()

        if not self.metadata['sklearn_version'] == sklearn.__version__:
            msg = ('The model was created with Scikit-Learn version {} '
                   'but version {} has been installed in your current environment.'
                  ).format(self.metadata['sklearn_version'], sklearn.__version__)
            raise RuntimeError(msg)


        with fsspec.open(self._urlpath, **self._storage_options) as f:
            return joblib.load(f)

