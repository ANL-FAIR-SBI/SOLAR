import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import math
from PIL import Image
import numpy as np
import time
from torchvision.datasets import VisionDataset


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetFolder1(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        nsamples,
        rank,
        size,
        cache_size,
        indices,
        to_load,
        local_batch_size,
        root: str,
        loader: Callable[[str], Any]=default_loader,
        extensions: Optional[Tuple[str, ...]] = IMG_EXTENSIONS,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.epoch=0
        self.num_call = 0
        self.step = 0
        self.nsamples = nsamples
        self.bench_load_step=set()
        self.cached_data_idx = dict()
        self.prefetch_buffer = dict()
        self.cache_size = cache_size
        self.loc_batch_size=local_batch_size
        self.rank = rank
        self.size = size
        self.idx_to_load=to_load
        self.idx_to_load_total=to_load
        self.idx_extra_load=set()
        self.loaded=dict()
        self.load_numbers = 0
        self.cache_load = 0
        self.indices = indices
        self.to_load_per_call = 0
        self.no_load_after_call = 0
        self.not_using=set()
        self.idx_extra_load_total = set()
        self.loaded_curr_step = set()
        self.load_time = 0
        self.cache_time = 0
        self.chunk_load_numbers = 0
        
    def getLoadNumber(self):
        return self.load_numbers
        #return len(self.loaded_curr_step)
    
    def getCacheLoad(self):
        return self.cache_load

    def get_time(self):
        print(self.load_time)
        return self.load_time,self.cache_time

    def getChunkLoad(self):
        return self.chunk_load_numbers

    def set_epoch(self,epoch):
        self.epoch = epoch
        self.num_call = 0
        self.load_time = 0
        self.cache_time = 0
    
    def set_step(self,step):
        self.step=step
        self.load_time = 0
        self.cache_time = 0
        self.load_numbers = 0
        self.chunk_load_numbers = 0
        self.cache_load = 0
        self.to_load_per_call = 0
        self.num_call = 0
        if self.epoch > 0 and self.step < len(self.idx_to_load[self.epoch-1]):
            self.not_using = self.idx_to_load[self.epoch-1][self.step]
        if self.epoch > 0:
            if len(self.idx_to_load[self.epoch-1]) > self.step:
                self.idx_extra_load = set(list(self.idx_to_load[self.epoch-1][self.step])[self.rank::self.size])
                self.idx_extra_load_total = set(list(self.idx_to_load[self.epoch-1][self.step]))
                self.to_load_per_call = 2
                self.no_load_after_call=math.floor(self.loc_batch_size-len(self.idx_extra_load)/self.to_load_per_call)
            else: 
                self.idx_extra_load = set()
        self.bench_load_step = set()

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def getItemBalancing(self,idx,flag):
        #self.loaded_curr_step.add(idx)
        cached=False
        prefetched=False
        if idx in self.cached_data_idx.keys():
            cached = True
        if idx in self.prefetch_buffer.keys():
            prefetched = True
        if not cached and not prefetched:
            #if not flag:
            self.load_numbers += 1
            load_time_start=time.perf_counter()
            path, target = self.samples[idx]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            self.load_time +=time.perf_counter()-load_time_start
            if flag:
                self.cached_data_idx[idx]=[sample, target]
                if len(self.cached_data_idx) > self.cache_size:
                    if 0 == self.epoch:
                        idx_to_replace=list(self.cached_data_idx.keys())[0]
                    else:
                        idx_to_replace=list(self.cached_data_idx.keys())[0]
                        if self.step < len(self.idx_to_load[self.epoch-1]):
                            for k in self.cached_data_idx.keys():
                                if k in self.not_using and k!=idx:
                                    idx_to_replace = k
                                    break
                    self.cached_data_idx.pop(idx_to_replace)
            else:
                self.prefetch_buffer[idx]=[sample, target]
            
        elif cached and not prefetched:
            self.cache_load += 1
            sample=self.cached_data_idx[idx][0]
            target=self.cached_data_idx[idx][1]
        elif prefetched and not cached:
            self.cache_load += 1
            sample=self.prefetch_buffer[idx][0]
            target=self.prefetch_buffer[idx][1]
        #print("!!!!!")
        return sample, target
    
    def getItemChunked(self,st,ed):
        self.chunk_load_numbers += 1
        #print("sss: %s, %s" %(st,ed))
        load_time_start=time.perf_counter()
        path, target = self.samples[st:ed]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        self.load_time +=time.perf_counter()-load_time_start
        #print('3:'+str(x.size()))
        self.prefetch_buffer[st]=[]
        self.prefetch_buffer[ed]=[x[:,-1,:,:],y1[:,-1,:,:],y2[:,-1,:,:]]
        
        return sample,target
    
    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        self.num_call += 1
        idx = int(self.indices[self.epoch][index])
        sample_list=[]
        target_list=[]
        if self.epoch == 0:
            return self.getItemBalancing(idx,True)
        if self.to_load_per_call > 0 and self.num_call <= self.no_load_after_call and self.epoch > 0:
            #self.idx_extra_load = list(self.idx_to_load[self.epoch-1][self.step])[rank::size]
            for tt in range(self.to_load_per_call):
                if len(self.idx_extra_load) > 0:
                    #t_idx = None
                    #for i in range(len(self.idx_extra_load)):
                    tt_idx = int(self.idx_extra_load.pop())
                    sample_temp, target_temp = self.getItemBalancing(tt_idx,False)
                    sample_list.append(sample_temp)
                    target_list.append(target_temp)
        #if idx not in self.loaded_curr_step:
        #if idx not in self.idx_extra_load_total:  
        sample, target = self.getItemBalancing(idx,True)
        sample_list.append(sample)
        target_list.append(target)

        return sample, target

    def __len__(self) -> int:
        return self.nsamples




class ImageFolder1(DatasetFolder1):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        nsamples,
        rank,
        size,
        cache_size,
        indices,
        to_load,
        local_batch_size,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        
    ):
        super().__init__(
            nsamples,
            rank,
            size,
            cache_size,
            indices,
            to_load,
            local_batch_size,
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples