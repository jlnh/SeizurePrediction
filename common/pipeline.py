class Pipeline(object):
    """
    A Pipeline is an object representing the data transformations to make
    on the input data, finally outputting extracted features.

    gen_preictal: Whether preictal data generation should be used for this pipeline

    pipeline: List of transforms to apply one by one to the input data
    """
    def __init__(self, gen_preictal, pipeline):
        self.transforms = pipeline
        self.gen_preictal = gen_preictal
        names = [t.get_name() for t in self.transforms]
        if gen_preictal:
            names = ['gen'] + names
        self.name = 'empty' if len(names) == 0 else '_'.join(names)

    def get_name(self):
        return self.name

    def apply(self, data):
        for transform in self.transforms:
            data = transform.apply(data)
        return data
