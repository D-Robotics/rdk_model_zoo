import { describe, expect, it } from 'vitest';
import { formatMetricValue } from '../src/catalog/metric-display';
import { accuracyValueText } from '../src/ui/accuracy-comparison';
import type { MetricRecord } from '../src/catalog/types';

describe('preserves source metric bounds in rendered values', () => {
  it('does not present the ConvNeXt 200+ FPS bound as an exact measurement', () => {
    const metric: MetricRecord = {metric:'throughput',value:200,unit:'fps',qualifier:'lower-bound',concurrency:4};
    expect(formatMetricValue(metric,'en',{asPercentage:false})).toMatch(/(?:≥|>=|\+|at least)/);
  });
  it('preserves an accuracy lower bound without rounding to an exact value', () => {
    const metric: MetricRecord = {metric:'cosine_similarity',value:0.9999,unit:'ratio',qualifier:'lower-bound'};
    expect(accuracyValueText(metric,'zh')).toMatch(/(?:≥|>=|\+|至少)/);
    expect(accuracyValueText(metric,'zh')).toContain('0.9999');
  });
});

describe('qualifier prefixes and source precision', () => {
  it('marks upper bounds and approximations instead of implying exact values', () => {
    const upper: MetricRecord = {metric:'throughput',value:15870.5,unit:'fps',qualifier:'upper-bound'};
    expect(formatMetricValue(upper,'en',{asPercentage:false})).toContain('≤');
    const approx: MetricRecord = {metric:'throughput',value:113.35,unit:'fps',qualifier:'approximate'};
    expect(formatMetricValue(approx,'en',{asPercentage:false})).toContain('≈');
  });
  it('renders exact and unqualified measurements without a prefix', () => {
    const exact: MetricRecord = {metric:'throughput',value:200,unit:'fps',qualifier:'exact'};
    expect(formatMetricValue(exact,'en',{asPercentage:false})).toBe('200fps');
    const unqualified: MetricRecord = {metric:'throughput',value:200,unit:'fps'};
    expect(formatMetricValue(unqualified,'en',{asPercentage:false})).toBe('200fps');
  });
  it('keeps all six fractional digits of the published cosine values', () => {
    for (const value of [0.968013, 0.965641, 0.997313]) {
      const metric: MetricRecord = {metric:'cosine_similarity',value,unit:'ratio',qualifier:'exact'};
      expect(accuracyValueText(metric,'en')).toBe(String(value));
      expect(accuracyValueText(metric,'zh')).toBe(String(value));
    }
  });
  it('combines the bound prefix with the unrounded source precision', () => {
    const lower: MetricRecord = {metric:'cosine_similarity',value:0.965641,unit:'ratio',qualifier:'lower-bound'};
    expect(accuracyValueText(lower,'en')).toBe('≥0.965641');
    const upper: MetricRecord = {metric:'cosine_similarity',value:0.997313,unit:'ratio',qualifier:'upper-bound'};
    expect(accuracyValueText(upper,'en')).toBe('≤0.997313');
    const approx: MetricRecord = {metric:'cosine_similarity',value:0.968013,unit:'ratio',qualifier:'approximate'};
    expect(accuracyValueText(approx,'zh')).toBe('≈0.968013');
  });
  it('still honors an explicit maximumFractionDigits option', () => {
    const metric: MetricRecord = {metric:'top-1',value:76.123456,unit:'percent',qualifier:'approximate'};
    expect(formatMetricValue(metric,'en')).toBe('≈76.12%');
    expect(formatMetricValue(metric,'en',{maximumFractionDigits:4})).toBe('≈76.1235%');
  });
});
