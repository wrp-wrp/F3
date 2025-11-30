use crate::File::fff::flatbuf as fb;

/// Metadata for a block-level micro-index (e.g., mini-IVF/PQ) embedded in the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicroIndex {
    pub wasm_offset: u64,
    pub wasm_size: u32,
    pub abi_major: u16,
    pub abi_minor: u16,
    pub aux_offset: u64,
    pub aux_size: u32,
    pub reserved: Vec<u32>,
}

impl MicroIndex {
    pub fn from_fb(node: fb::MicroIndex<'_>) -> Self {
        let reserved = node
            .reserved()
            .map(|v| v.iter().collect())
            .unwrap_or_default();
        Self {
            wasm_offset: node.wasm_offset(),
            wasm_size: node.wasm_size(),
            abi_major: node.abi_major(),
            abi_minor: node.abi_minor(),
            aux_offset: node.aux_offset(),
            aux_size: node.aux_size(),
            reserved,
        }
    }
}

/// Block-level layout information for vector columns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorBlock {
    pub offset: u64,
    pub size: u32,
    pub row_start: u64,
    pub row_count: u32,
    pub align: u32,
    pub micro_index: MicroIndex,
}

impl VectorBlock {
    pub fn from_fb(node: fb::VectorBlock<'_>) -> Self {
        let mi = node
            .micro_index()
            .map(MicroIndex::from_fb)
            .unwrap_or(MicroIndex {
                wasm_offset: 0,
                wasm_size: 0,
                abi_major: 0,
                abi_minor: 0,
                aux_offset: 0,
                aux_size: 0,
                reserved: Vec::new(),
            });
        Self {
            offset: node.offset(),
            size: node.size_(),
            row_start: node.row_start(),
            row_count: node.row_count(),
            align: node.align(),
            micro_index: mi,
        }
    }
}

/// Optional per-column vector metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorColumn {
    pub blocks: Vec<VectorBlock>,
    pub io_hint_bytes: u32,
}

impl VectorColumn {
    pub fn from_fb(node: fb::VectorColumn<'_>) -> Self {
        let blocks = node
            .blocks()
            .map(|v| v.iter().map(VectorBlock::from_fb).collect())
            .unwrap_or_default();
        Self {
            blocks,
            io_hint_bytes: node.io_hint_bytes(),
        }
    }
}
