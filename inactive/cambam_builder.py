# New Cambam Builder module
import xml.etree.ElementTree as ET
import os
import pickle
import uuid
import logging
import math
import numpy as np
from typing import (
    List, Dict, Tuple, Union, Optional, Set, Any, Sequence, cast, Type, TypeVar
)

# Module imports
from cad_transformations import *

# Logger
logger = logging.getLogger(__name__)
# if logging is not configured elsewhere:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


