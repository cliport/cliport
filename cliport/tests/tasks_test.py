"""Integration tests for dvnets tasks."""

from absl.testing import absltest
from absl.testing import parameterized
from cliport import tasks
from cliport.environments import environment

ASSETS_PATH = 'cliport/environments/assets/'


class TaskTest(parameterized.TestCase):

    def _create_env(self):
        assets_root = ASSETS_PATH
        env = environment.Environment(assets_root)
        env.seed(0)
        return env

    def _run_oracle_in_env(self, env):
        agent = env.task.oracle(env)
        obs = env.reset()
        info = None
        done = False
        for _ in range(10):
            act = agent.act(obs, info)
            obs, _, done, info = env.step(act)
            if done:
                break

    @parameterized.named_parameters((
    # demo conditioned
            'AlignBoxCorner',
            tasks.AlignBoxCorner(),
    ), (
            'AssemblingKits',
            tasks.AssemblingKits(),
    ), (
            'AssemblingKitsEasy',
            tasks.AssemblingKitsEasy(),
    ), (
            'BlockInsertion',
            tasks.BlockInsertion(),
    ), (
            'ManipulatingRope',
            tasks.ManipulatingRope(),
    ), (
            'PackingBoxes',
            tasks.PackingBoxes(),
    ), (
            'PalletizingBoxes',
            tasks.PalletizingBoxes(),
    ), (
            'PlaceRedInGreen',
            tasks.PlaceRedInGreen(),
    ), (
            'StackBlockPyramid',
            tasks.StackBlockPyramid(),
    ), (
            'SweepingPiles',
            tasks.SweepingPiles(),
    ), (
            'TowersOfHanoi',
            tasks.TowersOfHanoi(),

    # goal conditioned
    ), (
            'AlignRope',
            tasks.AlignRope(),
    ), (
            'AssemblingKitsSeqSeenColors',
            tasks.AssemblingKitsSeqSeenColors(),
    ), (
            'AssemblingKitsSeqUnseenColors',
            tasks.AssemblingKitsSeqUnseenColors(),
    ), (
            'AssemblingKitsSeqFull',
            tasks.AssemblingKitsSeqFull(),
    ), (
            'PackingShapes',
            tasks.PackingShapes(),
    ), (
            'PackingBoxesPairsSeenColors',
            tasks.PackingBoxesPairsSeenColors(),
    ), (
            'PackingBoxesPairsUnseenColors',
            tasks.PackingBoxesPairsUnseenColors(),
    ), (
            'PackingBoxesPairsFull',
            tasks.PackingBoxesPairsFull(),
    ), (
            'PackingSeenGoogleObjectsSeq',
            tasks.PackingSeenGoogleObjectsSeq(),
    ), (
            'PackingUnseenGoogleObjectsSeq',
            tasks.PackingUnseenGoogleObjectsSeq(),
    ), (
            'PackingSeenGoogleObjectsGroup',
            tasks.PackingSeenGoogleObjectsGroup(),
    ), (
            'PackingUnseenGoogleObjectsGroup',
            tasks.PackingUnseenGoogleObjectsGroup(),
    ), (
            'PutBlockInBowlSeenColors',
            tasks.PutBlockInBowlSeenColors(),
    ), (
            'PutBlockInBowlUnseenColors',
            tasks.PutBlockInBowlUnseenColors(),
    ), (
            'PutBlockInBowlFull',
            tasks.PutBlockInBowlFull(),
    ), (
            'StackBlockPyramidSeqSeenColors',
            tasks.StackBlockPyramidSeqSeenColors(),
    ), (
            'StackBlockPyramidSeqUnseenColors',
            tasks.StackBlockPyramidSeqUnseenColors(),
    ), (
            'StackBlockPyramidSeqFull',
            tasks.StackBlockPyramidSeqFull(),
    ), (
            'SeparatingPilesSeenColors',
            tasks.SeparatingPilesUnseenColors(),
    ), (
            'SeparatingPilesUnseenColors',
            tasks.SeparatingPilesUnseenColors(),
    ), (
            'SeparatingPilesFull',
            tasks.SeparatingPilesFull(),
    ), (
            'TowersOfHanoiSeqSeenColors',
            tasks.TowersOfHanoiSeqSeenColors(),
    ), (
            'TowersOfHanoiSeqUnseenColors',
            tasks.TowersOfHanoiSeqUnseenColors(),
    ), (
            'TowersOfHanoiSeqFull',
            tasks.TowersOfHanoiSeqFull(),
    ))
    def test_all_tasks(self, dvnets_task):
        env = self._create_env()
        env.set_task(dvnets_task)
        self._run_oracle_in_env(env)


if __name__ == '__main__':
    absltest.main()
