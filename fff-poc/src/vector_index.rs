use crate::file::footer::MetadataSection;
use fff_format::File::fff::flatbuf as fb;
use fff_format::ToFlatBuffer;
use flatbuffers::{FlatBufferBuilder, WIPOffset};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorIndexAlgorithm {
    BruteForce,
    Hnsw,
    Ivf,
    CustomWasm,
}

impl From<VectorIndexAlgorithm> for fb::VectorIndexAlgorithm {
    fn from(value: VectorIndexAlgorithm) -> Self {
        match value {
            VectorIndexAlgorithm::BruteForce => fb::VectorIndexAlgorithm::BRUTE_FORCE,
            VectorIndexAlgorithm::Hnsw => fb::VectorIndexAlgorithm::HNSW,
            VectorIndexAlgorithm::Ivf => fb::VectorIndexAlgorithm::IVF,
            VectorIndexAlgorithm::CustomWasm => fb::VectorIndexAlgorithm::CUSTOM_WASM,
        }
    }
}

impl From<fb::VectorIndexAlgorithm> for VectorIndexAlgorithm {
    fn from(value: fb::VectorIndexAlgorithm) -> Self {
        match value {
            fb::VectorIndexAlgorithm::HNSW => VectorIndexAlgorithm::Hnsw,
            fb::VectorIndexAlgorithm::IVF => VectorIndexAlgorithm::Ivf,
            fb::VectorIndexAlgorithm::CUSTOM_WASM => VectorIndexAlgorithm::CustomWasm,
            _ => VectorIndexAlgorithm::BruteForce,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VectorDistanceMetric {
    L2,
    Cosine,
    Dot,
}

impl From<VectorDistanceMetric> for fb::VectorDistanceMetric {
    fn from(value: VectorDistanceMetric) -> Self {
        match value {
            VectorDistanceMetric::L2 => fb::VectorDistanceMetric::L2,
            VectorDistanceMetric::Cosine => fb::VectorDistanceMetric::COSINE,
            VectorDistanceMetric::Dot => fb::VectorDistanceMetric::DOT,
        }
    }
}

impl From<fb::VectorDistanceMetric> for VectorDistanceMetric {
    fn from(value: fb::VectorDistanceMetric) -> Self {
        match value {
            fb::VectorDistanceMetric::COSINE => VectorDistanceMetric::Cosine,
            fb::VectorDistanceMetric::DOT => VectorDistanceMetric::Dot,
            _ => VectorDistanceMetric::L2,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantizationMethod {
    None,
    Scalar,
    Product,
    Custom,
}

impl Default for QuantizationMethod {
    fn default() -> Self {
        QuantizationMethod::None
    }
}

impl From<QuantizationMethod> for fb::VectorQuantizationMethod {
    fn from(value: QuantizationMethod) -> Self {
        match value {
            QuantizationMethod::None => fb::VectorQuantizationMethod::NONE,
            QuantizationMethod::Scalar => fb::VectorQuantizationMethod::SCALAR,
            QuantizationMethod::Product => fb::VectorQuantizationMethod::PRODUCT,
            QuantizationMethod::Custom => fb::VectorQuantizationMethod::CUSTOM,
        }
    }
}

impl From<fb::VectorQuantizationMethod> for QuantizationMethod {
    fn from(value: fb::VectorQuantizationMethod) -> Self {
        match value {
            fb::VectorQuantizationMethod::SCALAR => QuantizationMethod::Scalar,
            fb::VectorQuantizationMethod::PRODUCT => QuantizationMethod::Product,
            fb::VectorQuantizationMethod::CUSTOM => QuantizationMethod::Custom,
            _ => QuantizationMethod::None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuantizationSegment {
    pub start_dim: u32,
    pub end_dim: u32,
    pub method: QuantizationMethod,
    pub bits: u8,
    pub params: Vec<u8>,
}

impl QuantizationSegment {
    fn to_fb<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<fb::QuantizationSegment<'fb>> {
        let params = if self.params.is_empty() {
            None
        } else {
            Some(fbb.create_vector(&self.params))
        };
        fb::QuantizationSegment::create(
            fbb,
            &fb::QuantizationSegmentArgs {
                start_dim: self.start_dim,
                end_dim: self.end_dim,
                method: self.method.clone().into(),
                bits: self.bits,
                params,
            },
        )
    }
}

impl From<&fb::QuantizationSegment<'_>> for QuantizationSegment {
    fn from(segment: &fb::QuantizationSegment<'_>) -> Self {
        Self {
            start_dim: segment.start_dim(),
            end_dim: segment.end_dim(),
            method: segment.method().into(),
            bits: segment.bits(),
            params: segment
                .params()
                .map(|p| p.bytes().to_vec())
                .unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QuantizationSpec {
    pub dimension: u32,
    pub segments: Vec<QuantizationSegment>,
}

impl QuantizationSpec {
    fn to_fb<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> Option<WIPOffset<fb::QuantizationSpec<'fb>>> {
        if self.dimension == 0 && self.segments.is_empty() {
            return None;
        }
        let segments = self
            .segments
            .iter()
            .map(|segment| segment.to_fb(fbb))
            .collect::<Vec<_>>();
        let segments_vec = if segments.is_empty() {
            None
        } else {
            Some(fbb.create_vector(&segments))
        };
        Some(fb::QuantizationSpec::create(
            fbb,
            &fb::QuantizationSpecArgs {
                dimension: self.dimension,
                segments: segments_vec,
            },
        ))
    }
}

impl From<&fb::QuantizationSpec<'_>> for QuantizationSpec {
    fn from(spec: &fb::QuantizationSpec<'_>) -> Self {
        Self {
            dimension: spec.dimension(),
            segments: spec
                .segments()
                .map(|segments| {
                    segments
                        .iter()
                        .map(|s| QuantizationSegment::from(&s))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VectorIndexDescriptor {
    pub index_id: u32,
    pub column: String,
    pub algorithm: VectorIndexAlgorithm,
    pub distance_metric: VectorDistanceMetric,
    pub priority: u16,
    pub usage_hint: Option<String>,
    pub quantization: QuantizationSpec,
    pub custom_params: Vec<u8>,
    pub data_section: Option<MetadataSection>,
    pub wasm_section: Option<MetadataSection>,
}

impl VectorIndexDescriptor {
    pub fn data_section(&self) -> Option<&MetadataSection> {
        self.data_section.as_ref()
    }

    pub fn wasm_section(&self) -> Option<&MetadataSection> {
        self.wasm_section.as_ref()
    }

    pub fn to_fb<'fb>(
        &self,
        fbb: &mut FlatBufferBuilder<'fb>,
    ) -> WIPOffset<fb::VectorIndexDescriptor<'fb>> {
        let column = fbb.create_string(&self.column);
        let usage_hint = self.usage_hint.as_ref().map(|hint| fbb.create_string(hint));
        let custom_params = if self.custom_params.is_empty() {
            None
        } else {
            Some(fbb.create_vector(&self.custom_params))
        };
        let quantization = self.quantization.to_fb(fbb);
        let data_section = self.data_section.as_ref().map(|section| section.to_fb(fbb));
        let wasm_section = self.wasm_section.as_ref().map(|section| section.to_fb(fbb));
        let args = fb::VectorIndexDescriptorArgs {
            index_id: self.index_id,
            column: Some(column),
            algorithm: self.algorithm.clone().into(),
            metric: self.distance_metric.clone().into(),
            priority: self.priority,
            usage_hint,
            data_section,
            wasm_section,
            quantization,
            custom_params,
        };
        fb::VectorIndexDescriptor::create(fbb, &args)
    }
}

impl<'a> From<&fb::VectorIndexDescriptor<'a>> for VectorIndexDescriptor {
    fn from(desc: &fb::VectorIndexDescriptor<'a>) -> Self {
        Self {
            index_id: desc.index_id(),
            column: desc.column().unwrap_or_default().to_string(),
            algorithm: desc.algorithm().into(),
            distance_metric: desc.metric().into(),
            priority: desc.priority(),
            usage_hint: desc.usage_hint().map(|s| s.to_string()),
            quantization: desc
                .quantization()
                .map(|q| QuantizationSpec::from(&q))
                .unwrap_or_default(),
            custom_params: desc
                .custom_params()
                .map(|p| p.bytes().to_vec())
                .unwrap_or_default(),
            data_section: desc.data_section().map(|sec| MetadataSection::from(&sec)),
            wasm_section: desc.wasm_section().map(|sec| MetadataSection::from(&sec)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct VectorIndexConfig {
    pub index_id: u32,
    pub column: String,
    pub algorithm: VectorIndexAlgorithm,
    pub distance_metric: VectorDistanceMetric,
    pub priority: u16,
    pub usage_hint: Option<String>,
    pub quantization: QuantizationSpec,
    pub custom_params: Vec<u8>,
    pub data: Vec<u8>,
    pub wasm_module: Option<Vec<u8>>,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            index_id: 0,
            column: String::new(),
            algorithm: VectorIndexAlgorithm::BruteForce,
            distance_metric: VectorDistanceMetric::L2,
            priority: 0,
            usage_hint: None,
            quantization: QuantizationSpec::default(),
            custom_params: Vec::new(),
            data: Vec::new(),
            wasm_module: None,
        }
    }
}

impl VectorIndexConfig {
    pub fn into_descriptor(
        self,
        data_section: Option<MetadataSection>,
        wasm_section: Option<MetadataSection>,
    ) -> VectorIndexDescriptor {
        VectorIndexDescriptor {
            index_id: self.index_id,
            column: self.column,
            algorithm: self.algorithm,
            distance_metric: self.distance_metric,
            priority: self.priority,
            usage_hint: self.usage_hint,
            quantization: self.quantization,
            custom_params: self.custom_params,
            data_section,
            wasm_section,
        }
    }
}
