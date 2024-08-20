import random
import math
import unittest

def mock_top_k_uids(available_uids:dict, incentives:list[float], k:int=30, alpha:float=0.7):
    assert (alpha<=1) and (alpha>0.5), ValueError("For the get_top_k_uids method, alpha needs to be between 0.5 and 1")
    # get available_uids
    available_uids_and_incentives = [(uid, incentives[uid]) for uid in available_uids.keys()]
    sorted_axon_list = sorted(available_uids_and_incentives, key=lambda x: x[1], reverse=True)
    # query a random sample of half of the top 10% of miners:
    top_k_axons = sorted_axon_list[:min(len(sorted_axon_list),k)]
    if len(sorted_axon_list) > k:
        top_n = math.ceil(k*alpha)
        bottom_remainder = math.floor(k*(1-alpha))
        bottom_trim = min(0, len(sorted_axon_list[k:])-bottom_remainder) # check that bottom_remainder is smaller than number of remaining values
        top_n -= bottom_trim # substract the negative value
        bottom_remainder += bottom_trim # add the negative value
        assert (top_n>0) and (bottom_remainder>=0), ValueError(f'Invalid call values: calling {top_n} top miners and {bottom_remainder} bottom miners')
        other_axons = [x[0] for x in random.sample(sorted_axon_list[k:], bottom_remainder)]
        random_top_axons = [x[0] for x in random.sample(top_k_axons, top_n)]
        selected_uids = random_top_axons + other_axons
        return {uid: available_uids[uid] for uid in selected_uids}
    else:
        return available_uids

class TestRewards(unittest.TestCase):
    
    def setUp(self):
        self.incentives = [1/(i + 1) for i in range(256)]
        self.available_uids = {i: str(i) for i in range(256)}  # UIDs with their string representation

    def test_no_available_uids(self):
        available_uids = {}
        selected_uids = mock_top_k_uids(available_uids, self.incentives, k=30, alpha=0.7)
        self.assertEqual(selected_uids, {})

    def test_available_uids_less_than_k(self):
        available_uids = {i: str(i) for i in range(20)}
        selected_uids = mock_top_k_uids(available_uids, self.incentives, k=30, alpha=0.7)
        self.assertEqual(len(selected_uids), len(available_uids))
        self.assertTrue(all(uid in available_uids for uid in selected_uids))

    def test_available_uids_greater_than_k_alpha_1(self):
        available_uids = {i: str(i) for i in range(100)}
        selected_uids = mock_top_k_uids(available_uids, self.incentives, k=30, alpha=1)
        self.assertEqual(len(selected_uids), 30)
        expected_top_uids = list(available_uids.keys())[:30]
        self.assertTrue(all(uid in expected_top_uids for uid in selected_uids))

    def test_available_uids_greater_than_k_alpha_float(self):
        available_uids = {i: str(i) for i in range(100)}
        selected_uids = mock_top_k_uids(available_uids, self.incentives, k=30, alpha=0.7123)
        top_n = math.ceil(30 * 0.7123)
        self.assertEqual(len(selected_uids), 30)
        top_uids = list(available_uids.keys())[:30]
        bottom_uids = list(available_uids.keys())[30:]
        self.assertTrue(all(uid in top_uids for uid in sorted(selected_uids.keys())[:top_n]))
        self.assertTrue(all(uid in bottom_uids for uid in sorted(selected_uids.keys())[top_n:]))
    
    # Additional Test Cases for Invalid Alpha Values
    def test_alpha_less_than_05(self):
        available_uids = {i: str(i) for i in range(100)}
        with self.assertRaises(AssertionError):  # AssertionError is raised due to the assert statement
            mock_top_k_uids(available_uids, self.incentives, k=30, alpha=0.4)

    def test_alpha_equal_to_05(self):
        available_uids = {i: str(i) for i in range(100)}
        with self.assertRaises(AssertionError):  # AssertionError is raised due to the assert statement
            mock_top_k_uids(available_uids, self.incentives, k=30, alpha=0.5)

    def test_alpha_greater_than_1(self):
        available_uids = {i: str(i) for i in range(100)}
        with self.assertRaises(AssertionError):  # AssertionError is raised due to the assert statement
            mock_top_k_uids(available_uids, self.incentives, k=30, alpha=1.1)

if __name__ == '__main__':
    unittest.main()
