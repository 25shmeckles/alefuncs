
import pytest
import numpy as np
from alefuncs import *


def test_future_value():
    assert future_value(interest=0.1,period=1,cash=100) == 110.0
    assert future_value(interest=0.1,period=2,cash=100) == 121.0
    with pytest.raises(ValueError) as excinfo:
        future_value(interest=-1,period=1,cash=100)
    assert '"interest" must be a float between 0 and 1' in str(excinfo.value)


def test_split_overlap():
    assert split_overlap(iterable=range(10),size=3,overlap=2) == [range(0, 3), range(1, 4), range(2, 5), range(3, 6), range(4, 7), range(5, 8), range(6, 9), range(7, 10)]
    assert split_overlap(iterable=list(range(10)),size=3,overlap=2) == [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]
    
    with pytest.raises(ValueError) as excinfo:
        split_overlap(iterable=range(10),size=-1,overlap=2)
    

def test_stamp_to_date():
    stamp = 1477558868.93
    assert str(stamp_to_date(stamp,time='utc')) == '2016-10-27 09:01:08.930000'
    assert str(stamp_to_date(int(stamp),time='utc')) == '2016-10-27 09:01:08'
    assert str(stamp_to_date(stamp,time='local')) == '2016-10-27 11:01:08.930000'

def test_cluster_patterns():
    a  = [1,2,3,4,5,6,5,4,3,2,1]
    a1 = [n+1 for n in a]
    a2 = [n+5 for n in a]
    a3 = [n+6 for n in a]
    patterns = [a,a1,a2,a3]
    assert cluster_patterns(patterns,t=2) == {0: [1], 1: [0], 2: [3], 3: [2]}
    assert cluster_patterns(patterns,t=5) == {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2]}
    assert cluster_patterns(patterns,t=0.2) == {0: [], 1: [], 2: [], 3: []}

def test_compare_patterns():
    a  = np.array([1,2,3,4,5,6,5,4,3,2,1])
    a1 = np.array([n+0.1 for n in a])
    a2 = np.array([n+1 for n in a])
    a3 = np.array([n+10 for n in a])
    assert compare_patterns(a,a) == 99.999999999
    assert compare_patterns(a,a1) == 95.69696969696969
    assert compare_patterns(a,a2) == 56.96969696969697
    assert compare_patterns(a2,a) == 72.33766233766234
    assert compare_patterns(center(a),center(a2)) == 99.999999999999943
    assert compare_patterns(a,a3) == -330.3030303030303

def test_center():
    array = np.array([681.7, 682.489, 681.31, 682.001, 682.001, 682.499, 682.001])
    np.testing.assert_array_equal(center(array), 
                                  np.array([-0.30014285714287325, 0.48885714285711401,
                                            -0.6901428571429733, 0.0008571428570576245,
                                             0.0008571428570576245, 0.49885714285710492,
                                             0.0008571428570576245]))

def test_rescale():
    a =  np.array([1,2,3,4,5,6,5,4,3,2,1])
    a1 = np.array([n*0.001 for n in a])
    a2 = np.array([n*10 for n in a])
    np.testing.assert_array_equal(rescale(a), np.array([0.0, 0.20000000000000001, 0.40000000000000002,
                                                               0.59999999999999998, 0.80000000000000004, 1.0, 
                                                               0.80000000000000004, 0.59999999999999998, 
                                                               0.40000000000000002, 0.20000000000000001, 0.0]))
    np.testing.assert_array_equal(rescale(a), rescale(a1))
    np.testing.assert_array_equal(rescale(a), rescale(a2))

def test_standardize():
    a =  np.array([1,2,3,4,5,6,5,4,3,2,1])
    assert list(standardize(a)) == [-1.419904585617662, -0.79514656794589078, -0.17038855027411956, 0.45436946739765166,
                                     1.0791274850694228, 1.7038855027411941, 1.0791274850694228, 0.45436946739765166,
                                    -0.17038855027411956, -0.79514656794589078, -1.419904585617662]

def test_normalize():
    a =  np.array([1,2,3,4,5,6,5,4,3,2,1])
    assert list(normalize(a)) == [0.082760588860236795, 0.16552117772047359, 0.24828176658071038, 0.33104235544094718,
                                  0.41380294430118397, 0.49656353316142077, 0.41380294430118397, 0.33104235544094718,
                                  0.24828176658071038, 0.16552117772047359, 0.082760588860236795]

def test_delta_percent():
    assert delta_percent(20,22) == 10.0
    assert delta_percent(2,20) == 900.0
    assert delta_percent(1,1) == 1e-09
    assert delta_percent(10,9) == -10.0










