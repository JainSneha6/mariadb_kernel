""" Maintains a list of magic commands supported by the kernel """

# Copyright (c) MariaDB Foundation.
# Distributed under the terms of the Modified BSD License.

from mariadb_kernel.maria_magics.line import Line
from mariadb_kernel.maria_magics.df import DF
from mariadb_kernel.maria_magics.lsmagic import LSMagic
from mariadb_kernel.maria_magics.maria_magic import MariaMagic
from mariadb_kernel.maria_magics.bar import Bar
from mariadb_kernel.maria_magics.pie import Pie
from mariadb_kernel.maria_magics.delimiter import Delimiter
from mariadb_kernel.maria_magics.load import Load
from mariadb_kernel.maria_magics.ml_commands.data_cleaning.missing import Missing
from mariadb_kernel.maria_magics.ml_commands.data_cleaning.dropmissing import DropMissing
from mariadb_kernel.maria_magics.ml_commands.data_cleaning.stats import Stats
from mariadb_kernel.maria_magics.ml_commands.data_cleaning.fillmissing import FillMissing
from mariadb_kernel.maria_magics.ml_commands.data_cleaning.outliers import Outliers

def get():
    return {
        "line": Line,
        "bar": Bar,
        "pie": Pie,
        "df": DF,
        "lsmagic": LSMagic,
        "delimiter": Delimiter,
        "load": Load,
        "missing": Missing,
        "dropmissing": DropMissing,
        "stats": Stats,
        "fillmissing": FillMissing,
        "outliers": Outliers,
    }
