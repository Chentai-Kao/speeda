% - inFileName:  without ".wav" extension
% - outFileName: with ".wav" extension
% - speed:       the normal speed-up speed
function run(inFileName, outFileName, speed)

  close all;

  showPlot = true;
  visibility = 'off';
  if showPlot
    visibility = 'on';
  end
  %h = figure('visible', visibility); hold on;

  % Load music (as row vector)
  [original, Fs] = wavread([inFileName '.wav']);
  original = original(:, 1)';
  audioLength = length(original) / Fs * 1000; % audio length in ms.

  % Run harma syllable segmentation
  [syllableStarts, syllableEnds] = detectSyllables(original, Fs);
  
  % Round timestamp to align with video frames (30 fps)
  %[syllableStarts, syllableEnds] = alignToFrame(syllableStarts, syllableEnds);

  % Calculate instant syllable speed.
  voteDensity = calcVoteDensity(original, Fs, syllableStarts, syllableEnds);

  % Filter the syllable density by median.
  voteDensityMedian = calcVoteDensityMedian(voteDensity);
  
  % Find valleys for segmentation.
  segPoints = calcSegments(voteDensityMedian);

  % Merge segments.
  segments = mergeSegments(segPoints);

  % Calculate syllable density corresponding to each segment.
  syllableDensity = calcSyllableDensity(segments, syllableStarts);

  % calculate speed ratio
  [starts, ends, ratios] = calcRatios(segments, syllableDensity, speed,...
                                      audioLength);

  % output ratio to file
%  of = fopen(outFileName, 'w');
%  for m = 1:length(starts)
%    fprintf(of, '%d %d %.2f\n', starts(m), ends(m), ratios(m));
%  end
%  fclose(of);

  % plot
  if showPlot
    figure();
    subplot(3, 1, 1);
    plot(original);
    set(gca, 'XTick', []);
    %xlim([1, 7 * Fs]);
    ylabel('Amptitude');
    title('Audio Signal');

    %plot(voteDensity, 'g');
    subplot(3, 1, 2); hold on;
    plot(voteDensityMedian, 'LineWidth', 3);
    set(gca, 'XTick', []);
    %xlim([1, 7000]);
    %ylim([0, 3]);
    ylabel('Number of votes');
    title('Syllable voteDensity');
    %plot(segPoints, 2 * ones(1, length(segPoints)), 'bo');
    plot(syllableStarts * 1000, 4 * ones(1, length(syllableStarts)), 'go');
    plot(syllableEnds * 1000, 3 * ones(1, length(syllableEnds)), 'rx');
    tmp_x = [];
    tmp_y = [];
    for m = 1:length(starts)
      tmp_x = [tmp_x, starts(m), ends(m)];
      tmp_y = [tmp_y, ratios(m) * [1, 1]];
    end
    subplot(3, 1, 3);
    plot(tmp_x, tmp_y, 'LineWidth', 3);
    %xlim([1, 7000]);
    xlabel('Time (ms)');
    ylabel('Ratio');
    title('Speed-up ratio');
  end
end

% Return number of samples (rounded down to integer) that holds
% the desired time (in seconds).
function samples = time2samples(time, Fs)
  samples = floor(time * Fs);
end

% Append src to out. Keep track of the position of writing.
function [writeEnd, out] = appendOutput(src, out, writePos)
  writeEnd = writePos + length(src);
  out(writePos:writeEnd - 1) = src;
end

% Detect syllable using Harma method, return the start/end time of syllables.
% AUDIO is a row vector.
function [syllableStarts, syllableEnds] = detectSyllables(audio, Fs)
  [syllables, Fs, S, F, T, P] =...
      harmaSyllableSeg(audio', Fs, kaiser(256), 128, 256, 20);
  syllableStarts = zeros(1, length(syllables));
  syllableEnds = zeros(1, length(syllables));
  for m = 1:length(syllables)
    syllableStarts(m) = min(syllables(m).times);
    syllableEnds(m) = max(syllables(m).times);
  end
  syllableStarts = sort(syllableStarts);
  syllableEnds = sort(syllableEnds);
end

% Align syllable start/end timestamp to multiple of video frame. (30 fps)
function [starts, ends] = alignToFrame(starts, ends)
  for m = 1:length(starts)
    starts(m) = alignTimestamp(starts(m));
    ends(m) = alignTimestamp(ends(m));
  end
end

% Align a timestamp to multiple of video frame (30 fps)
function timestamp = alignTimestamp(timestamp)
  fps = 30;
  timestamp = round(timestamp * fps) / fps;
end
 
% Calculate instant syllable speed.
% Each syllable casts 1 vote to every nearby time frame (unit: 1ms)
% within voteWindow of start and end time of the syllable.
function voteDensity = calcVoteDensity(audio, Fs, syllableStarts, syllableEnds)
  voteWindow = 0.3; % in second
  voteDensity = zeros(1, floor(length(audio) / Fs * 1000)); % length in ms
  for m = 1:length(syllableStarts)
    % calculate the range to cast vote. Clamp to (1, end) respetively.
    voteStart = floor((syllableStarts(m) - voteWindow) * 1000);
    voteEnd = floor((syllableEnds(m) + voteWindow) * 1000);
    if voteStart < 1
      voteStart = 1;
    end
    if voteEnd > length(voteDensity)
      voteEnd = length(voteDensity);
    end
    % cast vote
    for n = voteStart:voteEnd
      voteDensity(n) = voteDensity(n) + 1;
    end
  end
end

function voteDensityMedian = calcVoteDensityMedian(voteDensity)
  windowSize = 151;
  voteDensityMedian = medfilt1(voteDensity, windowSize);
end

function segPoints = calcSegments(voteDensity)
  segPoints = [1];
  inValley = true;
  valleyStart = 1;
  for m = 2:length(voteDensity)
    if voteDensity(m - 1) < voteDensity(m)
      if inValley % valley ends
        segPoints = [segPoints, valleyStart, m];
      end
      inValley = false;
    elseif voteDensity(m - 1) > voteDensity (m)
      valleyStart = m;
      inValley = true;
    end
  end
  % Make sure 'segPoints' has the end point of 'voteDensity'
  if segPoints(end) ~= length(voteDensity)
    segPoints = [segPoints length(voteDensity)];
  end
end

function segments = mergeSegments(segPoints)
  minSegmentLength = 400; % unit: ms
  segments = [1];
  segStart = segPoints(1);
  for m = 2:length(segPoints)
    if segPoints(m) - segStart > minSegmentLength
      segments = [segments segPoints(m)];
      segStart = segPoints(m);
    end
  end
end

function [starts, ends, ratios] = calcRatios(segments, syllableDensity,...
                                             speed, audioLength)
  starts = [];
  ends = [];
  ratios = [];
  pauseTime = 150; % desired pause time length (unit: ms)
  avgDensity = mean(syllableDensity);
  disp(sprintf('average syllableDensity: %.3f', avgDensity))

  % pause count and speak time
  pauseCount = 0;
  speakTime = 0;
  for m = 2:length(segments)
    segStart = segments(m - 1);
    segEnd = segments(m) - 1; % TODO should this overlap?
    if syllableDensity(m - 1) == 0 && segEnd - segStart > pauseTime
      pauseCount = pauseCount + 1;
    else
      ratio = avgDensity / syllableDensity(m - 1);
      speakTime = speakTime + (segEnd - segStart) / ratio;
    end
  end
  % calculate desired ratio
  expectTime = audioLength / speed;
  desiredRatio = speakTime / (expectTime - pauseCount * pauseTime);
  % speed up
  for m = 2:length(segments)
    segStart = segments(m - 1);
    segEnd = segments(m) - 1; % TODO should this overlap?
    starts = [starts segStart];
    ends = [ends segEnd];
    % ratio
    if syllableDensity(m - 1) == 0 && segEnd - segStart > pauseTime
      ratio = (segEnd - segStart) / pauseTime;
    else
      ratio = avgDensity * desiredRatio / syllableDensity(m - 1);
    end
    ratios = [ratios ratio];
  end
end

% 'segments' starts from the first sample, ends at the last sample.
% e.g. for a music with 1000 audio samples, segments = [1, 23, ..., 1000]
% Therefore, there are k segments if length(segments) == k+1
% Note: unit of syllableStarts is second, segments is milli-second
function syllableDensity = calcSyllableDensity(segments, syllableStarts)
  syllableDensity = zeros(1, length(segments) - 1);
  index = 1; % syllable index
  for m = 2:length(segments)
    count = 0;
    while index <= length(syllableStarts) &&...
        (1000 * syllableStarts(index) < segments(m))
      index = index + 1;
      count = count + 1;
    end
    syllableDensity(m - 1) = count / (segments(m) - segments(m - 1));
  end
end
