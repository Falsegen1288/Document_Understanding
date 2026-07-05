import { PageData, Detection } from '../types';

export const INITIAL_PAGES: PageData[] = [
  {
    page: 7,
    displayName: 'CATALOG_PAGE_007_PILLING_INTRO.pdf',
    image_path: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBbZ8lTgfLxej6mZnnZPKG8LtG6RKCUt6CWPemkPGUM5VSZTzAKcK-Zy10op_-477C9MixAGQIS1MXgof_0o6OEE-IPiTejLTbtGNmO8umpVk9VBL-PZRcR1vQQX9EhW417Qcmrw4dXnKpzIngvWEund-TDP5HBz8N8SQENcdrrW-nFOt6nFjGQigFdcZ3RgWAdUJ_Y41z8Asxs0aCt3Dtn_RZocSHTXQWGQeZfsnX7BUyQXt56KMAxQrsWlxt8vzA7XkTVaqZpi-o',
    image_size: [1275, 1650],
    model_a: {
      name: 'DocLayoutYOLO',
      detections: [
        { id: '7-dl-0', type: 'title', bbox: [120, 80, 1150, 170], model: 'DocLayoutYOLO' },
        { id: '7-dl-1', type: 'section_header', bbox: [120, 190, 700, 230], model: 'DocLayoutYOLO' },
        { id: '7-dl-2', type: 'text', bbox: [120, 270, 600, 510], model: 'DocLayoutYOLO' },
        { id: '7-dl-3', type: 'text', bbox: [120, 530, 600, 840], model: 'DocLayoutYOLO' },
        { id: '7-dl-4', type: 'text', bbox: [650, 270, 1150, 490], model: 'DocLayoutYOLO' },
        { id: '7-dl-5', type: 'text', bbox: [650, 510, 1150, 740], model: 'DocLayoutYOLO' },
        { id: '7-dl-6', type: 'picture', bbox: [150, 870, 1120, 1545], model: 'DocLayoutYOLO' },
      ],
    },
    model_b: {
      name: 'Nemotron-Parse-v1.1',
      detections: [
        { id: '7-nm-0', type: 'title', bbox: [115, 82, 1160, 175], model: 'Nemotron-Parse-v1.1' },
        { id: '7-nm-1', type: 'text', bbox: [120, 270, 600, 510], model: 'Nemotron-Parse-v1.1' },
        { id: '7-nm-2', type: 'text', bbox: [120, 530, 600, 840], model: 'Nemotron-Parse-v1.1' },
        { id: '7-nm-3', type: 'text', bbox: [650, 270, 1150, 490], model: 'Nemotron-Parse-v1.1' },
        { id: '7-nm-4', type: 'picture', bbox: [150, 870, 1120, 1545], model: 'Nemotron-Parse-v1.1' },
      ],
    },
    model_c: {
      name: 'ADE-DPT2',
      detections: [
        { id: '7-ade-0', type: 'title', bbox: [120, 80, 1150, 170], model: 'ADE-DPT2' },
        { id: '7-ade-1', type: 'section_header', bbox: [120, 190, 700, 230], model: 'ADE-DPT2' },
        { id: '7-ade-2', type: 'text', bbox: [120, 270, 600, 510], model: 'ADE-DPT2' },
        { id: '7-ade-3', type: 'text', bbox: [120, 530, 600, 840], model: 'ADE-DPT2' },
        { id: '7-ade-4', type: 'text', bbox: [650, 270, 1150, 490], model: 'ADE-DPT2' },
        { id: '7-ade-5', type: 'text', bbox: [650, 510, 1150, 740], model: 'ADE-DPT2' },
        { id: '7-ade-6', type: 'picture', bbox: [150, 870, 1120, 1545], model: 'ADE-DPT2' },
      ],
    },
    ground_truth: [
      { id: '7-gt-0', type: 'title', bbox: [120, 80, 1150, 170], model: 'human' },
      { id: '7-gt-1', type: 'section_header', bbox: [120, 190, 700, 230], model: 'human' },
      { id: '7-gt-2', type: 'text', bbox: [120, 270, 600, 510], model: 'human' },
      { id: '7-gt-3', type: 'text', bbox: [120, 530, 600, 840], model: 'human' },
      { id: '7-gt-4', type: 'text', bbox: [650, 270, 1150, 490], model: 'human' },
      { id: '7-gt-5', type: 'text', bbox: [650, 510, 1150, 740], model: 'human' },
      { id: '7-gt-6', type: 'picture', bbox: [150, 870, 1120, 1545], model: 'human' },
    ],
  },
  {
    page: 15,
    displayName: 'CATALOG_PAGE_015_N_COMPASS.pdf',
    image_path: 'https://lh3.googleusercontent.com/aida-public/AB6AXuAZEYdnp8kKdXsBTU-1gRtk7HZJD4IFAr4XR7Am2bCb2lgSDKw3ioDV-fqW0wySBB1a80Lso01iTx8cItmvV5uEtEDrZdLeX2RIARVvIdDuPjhgIiI8RIEHEqZlS4gngScD3cwAOMl19e4LtfdMnXnOrMxbwXumWbNvV3yv4sHG9P0injsuBbxEPSbrdkSXb5xTv6HUsMnq48yiiXl2B10sZSM5BAMaRie0kIC1roaqDT-fsrxFkH6OpHN14oJJw_FnpIMsUEYwRNQ',
    image_size: [1275, 1650],
    model_a: {
      name: 'DocLayoutYOLO',
      detections: [
        { id: '15-dl-0', type: 'picture', bbox: [125, 115, 625, 440], model: 'DocLayoutYOLO' },
        { id: '15-dl-1', type: 'caption', bbox: [645, 145, 1140, 245], model: 'DocLayoutYOLO' },
        { id: '15-dl-2', type: 'picture', bbox: [650, 275, 1135, 740], model: 'DocLayoutYOLO' },
        { id: '15-dl-3', type: 'picture', bbox: [115, 375, 545, 790], model: 'DocLayoutYOLO' },
        { id: '15-dl-4', type: 'list_item', bbox: [120, 840, 590, 940], model: 'DocLayoutYOLO' },
        { id: '15-dl-5', type: 'list_item', bbox: [120, 960, 590, 1070], model: 'DocLayoutYOLO' },
        { id: '15-dl-6', type: 'list_item', bbox: [120, 1090, 590, 1210], model: 'DocLayoutYOLO' },
        { id: '15-dl-7', type: 'list_item', bbox: [645, 840, 1145, 1210], model: 'DocLayoutYOLO' },
      ],
    },
    model_b: {
      name: 'Nemotron-Parse-v1.1',
      detections: [
        { id: '15-nm-0', type: 'picture', bbox: [125, 115, 625, 440], model: 'Nemotron-Parse-v1.1' },
        { id: '15-nm-1', type: 'text', bbox: [645, 145, 1140, 245], model: 'Nemotron-Parse-v1.1' },
        { id: '15-nm-2', type: 'table', bbox: [650, 275, 1135, 740], model: 'Nemotron-Parse-v1.1' },
      ],
    },
    model_c: {
      name: 'ADE-DPT2',
      detections: [
        { id: '15-ade-0', type: 'picture', bbox: [125, 115, 625, 440], model: 'ADE-DPT2' },
        { id: '15-ade-1', type: 'caption', bbox: [645, 145, 1140, 245], model: 'ADE-DPT2' },
        { id: '15-ade-2', type: 'picture', bbox: [650, 275, 1135, 740], model: 'ADE-DPT2' },
        { id: '15-ade-3', type: 'list_item', bbox: [120, 840, 590, 940], model: 'ADE-DPT2' },
        { id: '15-ade-4', type: 'list_item', bbox: [120, 960, 590, 1070], model: 'ADE-DPT2' },
        { id: '15-ade-5', type: 'list_item', bbox: [645, 840, 1145, 1210], model: 'ADE-DPT2' },
      ],
    },
    ground_truth: null, // Touch-interactive
  },
  {
    page: 17,
    displayName: 'CATALOG_PAGE_017_SCALES_GAUGES.pdf',
    image_path: 'https://lh3.googleusercontent.com/aida-public/AB6AXuDyNPjgHSu9_XpH5UXjUTgnxYm3592GXlD7pBfMyul4CcUPUqE3fEHKJ3Cnr42G9QCv24tCeCw_1R1wQrsp7E1Rb8ECtOU8fpbm0SIToH_pdTTmJMaBecvc2TXU3SvOx0_q0Zevenri61sb2qvD9BFDFkXRw1YyWF_uDNjluhGzwfyEeSmzzA7Gvh7PlBZaLBDT25gaX32w88ckGZX4fZ9-WhVPS1NIg3v6avKLqNslUlLnUAR7klmdZjtUYagLKQT7LoFVgEqNB9U',
    image_size: [1275, 1650],
    model_a: {
      name: 'DocLayoutYOLO',
      detections: [
        { id: '17-dl-0', type: 'title', bbox: [120, 90, 1150, 175], model: 'DocLayoutYOLO' },
        { id: '17-dl-1', type: 'section_header', bbox: [120, 195, 600, 235], model: 'DocLayoutYOLO' },
        { id: '17-dl-2', type: 'table', bbox: [120, 275, 620, 1545], model: 'DocLayoutYOLO' },
        { id: '17-dl-3', type: 'table', bbox: [645, 275, 1145, 1545], model: 'DocLayoutYOLO' },
      ],
    },
    model_b: {
      name: 'Nemotron-Parse-v1.1',
      detections: [
        { id: '17-nm-0', type: 'title', bbox: [120, 90, 1150, 175], model: 'Nemotron-Parse-v1.1' },
        { id: '17-nm-1', type: 'table', bbox: [120, 275, 1145, 1545], model: 'Nemotron-Parse-v1.1' }, // Merged large single table
      ],
    },
    model_c: {
      name: 'ADE-DPT2',
      detections: [
        { id: '17-ade-0', type: 'title', bbox: [120, 90, 1150, 175], model: 'ADE-DPT2' },
        { id: '17-ade-1', type: 'section_header', bbox: [120, 195, 600, 235], model: 'ADE-DPT2' },
        { id: '17-ade-2', type: 'table', bbox: [120, 275, 620, 1545], model: 'ADE-DPT2' },
        { id: '17-ade-3', type: 'table', bbox: [645, 275, 1145, 1545], model: 'ADE-DPT2' },
      ],
    },
    ground_truth: null,
  },
  {
    page: 19,
    displayName: 'CATALOG_PAGE_019_MEASURING_GUIDE.pdf',
    image_path: 'https://lh3.googleusercontent.com/aida-public/AB6AXuBUVJUGygoEvob0hGsA7YadS2HJQUY2FyNXIt-aYtec6zUTPsgLpve4gTeeV3CSUdzkOM5nfTci8DEIOMZvMQyc-KaXMJMahcWeKcTU2-Ac3rpEJnwGxJwyk51cF4R7X-dKxDOssLxtmk8r7Yu4dNRF1S8szSWrgnzMlGSp5RH2Ve3IXNnZS9dOVcxOVwIkcCInoOn989Sci1lnYxC9ebJTqWm-QAuksZBDTTwBJOi44vmpxJcZdXr778sUmzNFV5rzaRcLRD2_Dw8',
    image_size: [1275, 1650],
    model_a: {
      name: 'DocLayoutYOLO',
      detections: [
        { id: '19-dl-0', type: 'title', bbox: [120, 95, 1150, 145], model: 'DocLayoutYOLO' },
        { id: '19-dl-1', type: 'picture', bbox: [120, 175, 600, 595], model: 'DocLayoutYOLO' },
        { id: '19-dl-2', type: 'picture', bbox: [645, 175, 1145, 595], model: 'DocLayoutYOLO' },
        { id: '19-dl-3', type: 'picture', bbox: [120, 645, 445, 895], model: 'DocLayoutYOLO' },
        { id: '19-dl-4', type: 'picture', bbox: [495, 645, 795, 895], model: 'DocLayoutYOLO' },
        { id: '19-dl-5', type: 'picture', bbox: [845, 645, 1145, 895], model: 'DocLayoutYOLO' },
        { id: '19-dl-6', type: 'section_header', bbox: [445, 915, 795, 955], model: 'DocLayoutYOLO' },
        { id: '19-dl-7', type: 'picture', bbox: [345, 975, 895, 1545], model: 'DocLayoutYOLO' },
      ],
    },
    model_b: {
      name: 'Nemotron-Parse-v1.1',
      detections: [
        { id: '19-nm-0', type: 'title', bbox: [120, 95, 1150, 145], model: 'Nemotron-Parse-v1.1' },
        { id: '19-nm-1', type: 'picture', bbox: [120, 175, 600, 595], model: 'Nemotron-Parse-v1.1' },
        { id: '19-nm-2', type: 'picture', bbox: [645, 175, 1145, 595], model: 'Nemotron-Parse-v1.1' },
        { id: '19-nm-3', type: 'picture', bbox: [120, 645, 1145, 895], model: 'Nemotron-Parse-v1.1' }, // Combined gallery picture
        { id: '19-nm-4', type: 'picture', bbox: [345, 975, 895, 1545], model: 'Nemotron-Parse-v1.1' },
      ],
    },
    model_c: {
      name: 'ADE-DPT2',
      detections: [
        { id: '19-ade-0', type: 'title', bbox: [120, 95, 1150, 145], model: 'ADE-DPT2' },
        { id: '19-ade-1', type: 'picture', bbox: [120, 175, 600, 595], model: 'ADE-DPT2' },
        { id: '19-ade-2', type: 'picture', bbox: [645, 175, 1145, 595], model: 'ADE-DPT2' },
        { id: '19-ade-3', type: 'picture', bbox: [120, 645, 445, 895], model: 'ADE-DPT2' },
        { id: '19-ade-4', type: 'picture', bbox: [495, 645, 795, 895], model: 'ADE-DPT2' },
        { id: '19-ade-5', type: 'picture', bbox: [845, 645, 1145, 895], model: 'ADE-DPT2' },
        { id: '19-ade-6', type: 'section_header', bbox: [445, 915, 795, 955], model: 'ADE-DPT2' },
        { id: '19-ade-7', type: 'picture', bbox: [345, 975, 895, 1545], model: 'ADE-DPT2' },
      ],
    },
    ground_truth: null,
  },
  {
    page: 25,
    displayName: 'CATALOG_PAGE_025_SPONGE_FORCEPS.pdf',
    image_path: 'https://lh3.googleusercontent.com/aida-public/AB6AXuCLQoVTbuffL6_rJTNhky1skQ_xf2KHFNEciQl4mik2qbRPyK2v6G6CSgRWUxE1brevhoAKE2TcSAaNMv-UnycRAAx6I5yNW37ifYTaEDXWXQHoD_mvuQr6tmgWR4lpCohC2OPHP6eL2FlqbkHlXykWCoE-78R9YG1uYT_z4j0s34AsW4S4r3-1_ZmkODNf-xDWyL2FOKTeNI_-AEVfGNbtjM1r1V2qHpxs6M6D92J6BOFFLgZe7K_tWcitEuJkBEKIQR5mzto7Vko',
    image_size: [1275, 1650],
    model_a: {
      name: 'DocLayoutYOLO',
      detections: [
        { id: '25-dl-0', type: 'title', bbox: [120, 95, 1150, 145], model: 'DocLayoutYOLO' },
        { id: '25-dl-1', type: 'caption', bbox: [120, 155, 400, 185], model: 'DocLayoutYOLO' },
        { id: '25-dl-2', type: 'section_header', bbox: [120, 215, 600, 250], model: 'DocLayoutYOLO' },
        { id: '25-dl-3', type: 'table', bbox: [445, 275, 1145, 375], model: 'DocLayoutYOLO' },
        { id: '25-dl-4', type: 'picture', bbox: [120, 275, 415, 415], model: 'DocLayoutYOLO' },
        { id: '25-dl-5', type: 'section_header', bbox: [120, 445, 600, 480], model: 'DocLayoutYOLO' },
        { id: '25-dl-6', type: 'table', bbox: [445, 515, 1145, 675], model: 'DocLayoutYOLO' },
        { id: '25-dl-7', type: 'picture', bbox: [120, 515, 415, 675], model: 'DocLayoutYOLO' },
        { id: '25-dl-8', type: 'section_header', bbox: [120, 705, 600, 740], model: 'DocLayoutYOLO' },
        { id: '25-dl-9', type: 'table', bbox: [445, 775, 1145, 875], model: 'DocLayoutYOLO' },
        { id: '25-dl-10', type: 'picture', bbox: [120, 775, 415, 875], model: 'DocLayoutYOLO' },
      ],
    },
    model_b: {
      name: 'Nemotron-Parse-v1.1',
      detections: [
        { id: '25-nm-0', type: 'title', bbox: [120, 95, 1150, 145], model: 'Nemotron-Parse-v1.1' },
        { id: '25-nm-1', type: 'table', bbox: [445, 275, 1145, 375], model: 'Nemotron-Parse-v1.1' },
        { id: '25-nm-2', type: 'table', bbox: [445, 515, 1145, 675], model: 'Nemotron-Parse-v1.1' },
        { id: '25-nm-3', type: 'table', bbox: [445, 775, 1145, 875], model: 'Nemotron-Parse-v1.1' },
      ],
    },
    model_c: {
      name: 'ADE-DPT2',
      detections: [
        { id: '25-ade-0', type: 'title', bbox: [120, 95, 1150, 145], model: 'ADE-DPT2' },
        { id: '25-ade-1', type: 'caption', bbox: [120, 155, 400, 185], model: 'ADE-DPT2' },
        { id: '25-ade-2', type: 'section_header', bbox: [120, 215, 600, 250], model: 'ADE-DPT2' },
        { id: '25-ade-3', type: 'table', bbox: [445, 275, 1145, 375], model: 'ADE-DPT2' },
        { id: '25-ade-4', type: 'picture', bbox: [120, 275, 415, 415], model: 'ADE-DPT2' },
        { id: '25-ade-5', type: 'section_header', bbox: [120, 445, 600, 480], model: 'ADE-DPT2' },
        { id: '25-ade-6', type: 'table', bbox: [445, 515, 1145, 675], model: 'ADE-DPT2' },
        { id: '25-ade-7', type: 'picture', bbox: [120, 515, 415, 675], model: 'ADE-DPT2' },
        { id: '25-ade-8', type: 'section_header', bbox: [120, 705, 600, 740], model: 'ADE-DPT2' },
        { id: '25-ade-9', type: 'table', bbox: [445, 775, 1145, 875], model: 'ADE-DPT2' },
        { id: '25-ade-10', type: 'picture', bbox: [120, 775, 415, 875], model: 'ADE-DPT2' },
      ],
    },
    ground_truth: null,
  },
];
