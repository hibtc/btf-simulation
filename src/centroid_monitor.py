"""
Monitor to save the centroid of the bunch 
adapted from the XSuite example

Author: Cristopher Cortés
Date: 2023-05-03
"""
import xtrack as xt
import xpart as xp
import xobjects as xo

from pathlib import Path


################################################################
# Definition of a beam element with an internal data recording #
################################################################

# We define a data structure to allow all elements of a new BeamElement type
# to store data in one place. Such a data structure needs to contain a field called
# `_index` of type `xtrack.RecordIndex`, which will be used internally to
# keep count of the number of records stored in the data structure. Together with
# the index, the structure can contain an arbitrary number of other fields (which
# need to be arrays) where the data will be stored.

class CentroidRecord(xo.Struct):

    at_turn = xo.Int64[:]
    x_cen = xo.Float64[:]
    y_cen = xo.Float64[:]
    
# To allow elements of a given type to store data in a structure of the type defined
# above we need to add in the element class an attribute called
# `_internal_record_class` to which we bind the data structure type defined above.

class CentroidMonitor(xt.BeamElement):
    _xofields={
        'start_at_turn':xo.Int64,
        'stop_at_turn' :xo.Int64,
        'data':CentroidRecord,
    }

    
    properties = [field.name for field in CentroidRecord._fields]

    _extra_c_sources = [Path(__file__).parent.absolute().joinpath('centroid_monitor.h')]

    
    def __init__(self, *, start_at_turn=None, stop_at_turn=None, _xobject=None, **kwargs):
        """
        Monitor to save the transversal centroid data of the tracked particles
        """
        if _xobject is not None:
            super().__init__(_xobject=_xobject)

        if start_at_turn is None:
            start_at_turn = 0
        if stop_at_turn is None:
            stop_at_turn = 0
        # explicitely init with zeros (instead of size only) to have consistent default values for untouched arrays
        # see also https://github.com/xsuite/xsuite/issues/294
        data = {prop: [0]*(stop_at_turn-start_at_turn) for prop in self.properties} # particle data
        super().__init__(start_at_turn=start_at_turn, stop_at_turn=stop_at_turn, data=data, **kwargs)
    

    def __getattr__(self, attr):
        if attr in self.properties:
            val = getattr(self.data, attr)
            val = val.to_nparray() # = self.data._buffer.context.nparray_from_context_array(val)
            return val
        return super().__getattr__(attr)


     
