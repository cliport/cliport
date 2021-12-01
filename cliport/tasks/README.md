# Tasks

### Descriptions

This folder contains a total of 10 goal-conditioned (language or image) and 8 demo-conditioned (original TransporterNets) tasks. 8 out of the 10 goal-conditioned tasks contain two splits: **seen** and **unseen**. The **full** version is a union of both **seen** and **unseen** attributes made specifically for multi-attr training. Sequential tasks that involve following instructions in a specific order are indicated by **seq** in their names.

See [__init__.py](__init__.py) for the full list of demo-conditioned and goal-conditioned (language or image) tasks:

```python
# demo conditioned
'align-box-corner': AlignBoxCorner,
'assembling-kits': AssemblingKits,
'assembling-kits-easy': AssemblingKitsEasy,
'block-insertion': BlockInsertion,
'block-insertion-easy': BlockInsertionEasy,
'block-insertion-nofixture': BlockInsertionNoFixture,
'block-insertion-sixdof': BlockInsertionSixDof,
'block-insertion-translation': BlockInsertionTranslation,
'manipulating-rope': ManipulatingRope,
'packing-boxes': PackingBoxes,
'palletizing-boxes': PalletizingBoxes,
'place-red-in-green': PlaceRedInGreen,
'stack-block-pyramid': StackBlockPyramid,
'sweeping-piles': SweepingPiles,
'towers-of-hanoi': TowersOfHanoi,

# goal conditioned
'align-rope': AlignRope,
'assembling-kits-seq-seen-colors': AssemblingKitsSeqSeenColors,
'assembling-kits-seq-unseen-colors': AssemblingKitsSeqUnseenColors,
'assembling-kits-seq-full': AssemblingKitsSeqFull,
'packing-shapes': PackingShapes,
'packing-boxes-pairs-seen-colors': PackingBoxesPairsSeenColors,
'packing-boxes-pairs-unseen-colors': PackingBoxesPairsUnseenColors,
'packing-boxes-pairs-full': PackingBoxesPairsFull,
'packing-seen-google-objects-seq': PackingSeenGoogleObjectsSeq,
'packing-unseen-google-objects-seq': PackingUnseenGoogleObjectsSeq,
'packing-seen-google-objects-group': PackingSeenGoogleObjectsGroup,
'packing-unseen-google-objects-group': PackingUnseenGoogleObjectsGroup,
'put-block-in-bowl-seen-colors': PutBlockInBowlSeenColors,
'put-block-in-bowl-unseen-colors': PutBlockInBowlUnseenColors,
'put-block-in-bowl-full': PutBlockInBowlFull,
'stack-block-pyramid-seq-seen-colors': StackBlockPyramidSeqSeenColors,
'stack-block-pyramid-seq-unseen-colors': StackBlockPyramidSeqUnseenColors,
'stack-block-pyramid-seq-full': StackBlockPyramidSeqFull,
'separating-piles-seen-colors': SeparatingPilesSeenColors,
'separating-piles-unseen-colors': SeparatingPilesUnseenColors,
'separating-piles-full': SeparatingPilesFull,
'towers-of-hanoi-seq-seen-colors': TowersOfHanoiSeqSeenColors,
'towers-of-hanoi-seq-unseen-colors': TowersOfHanoiSeqUnseenColors,
'towers-of-hanoi-seq-full': TowersOfHanoiSeqFull,
```



### Adding New Tasks

See [put_block_in_bowl.py](put_block_in_bowl.py) for an example on how a task is specified. Creating a new task involves: (1) setting up a scene with the desired objects, (2) specifying goals with a language instruction and target "zones" or "poses", (3) defining an evaluation metric that is either sequential or non-sequential. See the original [Ravens codebase](https://github.com/google-research/ravens) for more details on task specification and organization.

### Correcting COM for Google Scanned Objects

By default all [Google Scanned Objects](https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects) have COMs (Center of Mass) at the base of the object, which leads to weird behavior with the physics engine. To correct this, I manually edited the COM of each `.obj` file to be the geometric center of the mesh with [Blender](https://www.blender.org/). See this [guide on editing COMs](https://blender.stackexchange.com/questions/14294/how-to-recenter-an-objects-origin) for reference. After correction, the original `.obj` can be overwritten using Blender's Export option.

## Credit

All demo-conditioned are from [Ravens](https://github.com/google-research/ravens). The language-conditioned tasks were built-off the same PyBullet environments.