import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from flux_dcu.mapping import (ARMS, BACKEND, block_selection, schedule,
                              selection_statistics, split_sizes, wire_bytes)
from flux_dcu.planning import make_role, quantization_statistics


class MappingTests(unittest.TestCase):
    def test_common_entwine_mapping_and_local_spacing(self):
        for world in (2, 4, 8):
            base = schedule(256 * world, 4096, 512, world, 128, 'flux_taco')
            mixed = schedule(256 * world, 4096, 512, world, 128, 'flux_taco_interleaved')
            self.assertEqual(sorted(x['physical'] for x in mixed), list(range(2 * world)))
            self.assertEqual({(x['destination'], x['row']) for x in base},
                             {(x['destination'], x['row']) for x in mixed})
            for rank in range(world):
                slots = [x['slot'] for x in mixed if x['destination'] == rank]
                self.assertEqual(slots, [rank, rank + world])
            self.assertNotEqual(base, mixed)

    def test_collective_splits_match_every_pair(self):
        for world in (2, 4, 8):
            for destination in range(world):
                pairs = [split_sizes(world, rank, destination, 136) for rank in range(world)]
                for source in range(world):
                    for target in range(world):
                        self.assertEqual(pairs[source][0][target], pairs[target][1][source])
                self.assertEqual(sum(sum(s) for s, _ in pairs), (world - 1) * 136)

    def test_exact_byte_budget(self):
        self.assertEqual(wire_bytes(2048, 4096, 8, False), 14680064)
        self.assertEqual(wire_bytes(2048, 4096, 8, True), 7798784)

    def test_invalid_shapes_fail(self):
        for values in ((2049, 4096, 512, 8, 128), (2048, 130, 512, 8, 128),
                       (2048, 4096, 0, 8, 128), (2048, 4096, 512, 8, 512)):
            with self.assertRaises(ValueError):
                schedule(*values, 'flux')

    def test_arrival_plan_preserves_destinations_and_physical_lanes(self):
        for world in (2,4,8):
            shape=(256*world,4096,512)
            original=schedule(*shape,world,128,ARMS[2])
            reverse=[d*2+(1-within) for within in range(2) for d in range(world)]
            plan=make_role(*shape,world,128,order=reverse)
            changed=schedule(*shape,world,128,ARMS[3],plan)
            self.assertEqual([row['destination'] for row in changed],[i%world for i in range(2*world)])
            old_lanes={row['physical']:row['lane'] for row in original}
            self.assertEqual(old_lanes,{row['physical']:row['lane'] for row in changed})
            self.assertNotEqual([row['lane'] for row in original],[row['lane'] for row in changed])

    def test_explicit_plans_and_only_selective_arm_uses_mask(self):
        shape=(512,128,64)
        plan=make_role(*shape,2,128,mask=[False,True,False,False])
        self.assertEqual(block_selection(*shape,2,128,ARMS[3],plan),[True]*4)
        self.assertEqual(block_selection(*shape,2,128,ARMS[4],plan),plan['selective_mask'])
        for arm in ARMS[3:]:
            with self.assertRaises(ValueError): schedule(*shape,2,128,arm)
        for arm in ARMS[:3]:
            with self.assertRaises(ValueError): schedule(*shape,2,128,arm,plan)

    def test_plan_mutation_rejected(self):
        import copy
        shape=(512,128,64)
        original=make_role(*shape,2,128)
        for field,value in (('arrival_order',[0,0,1,3]),('arrival_order',[2,0,1,3]),
                            ('arrival_order',[False,2,1,3]),('selective_mask',[1,0,0,0]),
                            ('shape',[1024,128,64]),('chunk_rows',64),('diagnostics',None)):
            plan=copy.deepcopy(original);plan[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):
                schedule(*shape,2,128,ARMS[4],plan)

    def test_closed_block_byte_counts_match_independent_planner(self):
        for world in (2,4,8):
            m,n,k,chunk=256*world,4096,512,128
            for mask in ([False]*(2*world),[True]*(2*world),[i%3==0 for i in range(2*world)]):
                plan=make_role(m,n,k,world,chunk,mask=mask)
                actual=selection_statistics(m,n,world,chunk,mask)
                expected=quantization_statistics(plan,world)
                self.assertEqual(actual['remote_wire_bytes_by_source'],expected['remote_wire_bytes_per_source'])
                self.assertEqual(actual['selected_remote_blocks_by_source'],expected['selected_per_source'])
                self.assertEqual(sum(actual['selected_remote_blocks_by_source']),sum(mask)*(world-1))
                if all(mask): self.assertEqual(actual['remote_wire_bytes_by_source'],[wire_bytes(m,n,world,True)]*world)
                if not any(mask): self.assertEqual(actual['remote_wire_bytes_by_source'],[wire_bytes(m,n,world,False)]*world)
        self.assertEqual(BACKEND,'dcu-blas-rccl-block-lanes-v2')


if __name__ == '__main__':
    unittest.main()
