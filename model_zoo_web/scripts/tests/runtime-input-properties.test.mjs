import test from 'node:test';
import assert from 'node:assert/strict';
import {runtimeInputProperties} from '../runtime-input-properties.mjs';

test('converted featuremap pages display the actual declared dtype and layout', () => {
  const value = runtimeInputProperties({status:'converted',artifact:{},conversion:{runtime_input:{
    format:'FEATUREMAP',dtype:'float32',layout:'NCHW',shape:[1,3,512,512]}}}, {height:512,width:512});
  assert.equal(value.format,'featuremap');
  assert.equal(value.dtype,'float32');
  assert.equal(value.payload_layout,'NCHW');
  assert.deepEqual(value.resolution,[512,512]);
});

test('action models retain their temporal length alongside spatial resolution', () => {
  const value=runtimeInputProperties({status:'converted',artifact:{},conversion:{runtime_input:{
    format:'FEATUREMAP',dtype:'float32',layout:'NCTHW',shape:[1,3,16,112,112]}}}, {height:112,width:112});
  assert.equal(value.payload_layout,'NCTHW');
  assert.deepEqual(value.resolution,[16,112,112]);
});

test('existing published physical packing remains authoritative', () => {
  const value=runtimeInputProperties({status:'released',artifact:{runtime_input:'featuremap',
    runtime_tensor:{dtype:'float32',payload_layout:'NCHW',metadata_layout:'NHWC',packing:'recorded packing'}}}, {height:256,width:128});
  assert.equal(value.format,'featuremap');
  assert.equal(value.payload_layout,'NCHW');
  assert.equal(value.metadata_layout,'NHWC');
  assert.equal(value.packing,'recorded packing');
});
